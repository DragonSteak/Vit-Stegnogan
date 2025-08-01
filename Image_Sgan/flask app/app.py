from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
from PIL import Image, ImageFile
import io
import base64
from steganography_models import DenseEncoder, DenseDecoder
import torchvision.transforms as transforms
import traceback
import logging
import zlib
from reedsolo import RSCodec
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

# Initialize Reed-Solomon codec
rs = RSCodec(250)

def text_to_bits(text):
    """Convert text to a list of ints in {0, 1}"""
    return bytearray_to_bits(text_to_bytearray(text))

def bits_to_text(bits):
    """Convert a list of ints in {0, 1} to text"""
    return bytearray_to_text(bits_to_bytearray(bits))

def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray"""
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytearray(ints)

def text_to_bytearray(text):
    """Compress and add error correction"""
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))
    return x

def bytearray_to_text(x):
    """Apply error correction and decompress"""
    try:
        text = rs.decode(x)[0]
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except Exception as e:
        logger.error(f"Error in bytearray_to_text: {str(e)}")
        return ""

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

try:
    logger.info("Initializing encoder and decoder models...")
    encoder = DenseEncoder(data_depth=4, hidden_size=32).to(device)
    decoder = DenseDecoder(data_depth=4, hidden_size=32).to(device)

    # Load pretrained weights
    logger.info("Loading model weights...")
    encoder_state_dict = torch.load('encoder.pth', map_location=device)
    decoder_state_dict = torch.load('decoder.pth', map_location=device)
    
    # Load state dicts
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    logger.info("Model weights loaded successfully")

    encoder.eval()
    decoder.eval()
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def image_to_tensor(image):
    # Convert PIL image to tensor in [-1, 1] range
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # [0,1] -> [-1,1]
    ])
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor):
    # Convert tensor in [-1, 1] to PIL image in [0, 1]
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2  # [-1,1] -> [0,1]
    tensor = tensor.clamp(0, 1)
    transform = transforms.ToPILImage()
    return transform(tensor)

def text_to_tensor(text, size=(256, 256)):
    # Convert text to bits
    bits = text_to_bits(text)
    required_bits = size[0] * size[1] * 4
    if len(bits) < required_bits:
        bits = bits + [0] * (required_bits - len(bits))
    bits = bits[:required_bits]
    bits = np.array(bits, dtype=np.float32)
    bits = bits.reshape(1, 4, size[0], size[1])
    return torch.from_numpy(bits)

def tensor_to_text(tensor):
    # Convert decoder output to bits using threshold 0 (logits)
    binary = tensor.squeeze().cpu().numpy()
    logger.info(f"Raw decoder output statistics - min: {binary.min()}, max: {binary.max()}, mean: {binary.mean()}")
    
    # Try different thresholds
    thresholds = [0, -0.1, 0.1]
    best_result = ""
    best_candidates = Counter()
    
    for threshold in thresholds:
        logger.info(f"Trying threshold: {threshold}")
        binary_thresholded = binary > threshold
        bits = [1 if b else 0 for b in binary_thresholded.flatten()]
        logger.info(f"Number of 1s in binary output: {sum(bits)}, total bits: {len(bits)}")
        
        # Convert bits to bytearray
        bytearr = bits_to_bytearray(bits)
        logger.info(f"Bytearray length: {len(bytearr)}")
        
        # Split on b'\x00\x00\x00\x00' and decode candidates
        candidates = Counter()
        for candidate in bytearr.split(b'\x00\x00\x00\x00'):
            if len(candidate) < 4:  # Skip very short candidates
                continue
            try:
                msg = bytearray_to_text(bytearray(candidate))
                if msg:
                    candidates[msg] += 1
                    logger.info(f"Found valid candidate with threshold {threshold}: {msg}")
            except Exception as e:
                logger.error(f"Error processing candidate with threshold {threshold}: {str(e)}")
        
        if len(candidates) > len(best_candidates):
            best_candidates = candidates
            best_threshold = threshold
    
    if len(best_candidates) == 0:
        logger.error("No valid candidates found with any threshold")
        return ""
    
    candidate, count = best_candidates.most_common(1)[0]
    logger.info(f"Selected candidate with count {count} using threshold {best_threshold}: {candidate}")
    return candidate

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    try:
        logger.info("Received encode request")
        if 'image' not in request.files or 'text' not in request.form:
            logger.error("Missing image or text in request")
            return jsonify({'error': 'Missing image or text'}), 400
        
        # Get image and text
        image_file = request.files['image']
        text = request.form['text']
        logger.info(f"Processing encode request with text: {text}")
        
        # Process image
        image = Image.open(image_file)
        logger.info(f"Original image size: {image.size}")
        cover_tensor = image_to_tensor(image).to(device)
        logger.info(f"Cover tensor shape: {cover_tensor.shape}")
        
        # Process text
        secret_tensor = text_to_tensor(text).to(device)
        logger.info(f"Secret tensor shape: {secret_tensor.shape}")
        
        # Encode
        logger.info("Starting encoding process...")
        with torch.no_grad():
            stego_tensor = encoder(cover_tensor, secret_tensor)
        logger.info(f"Stego tensor shape: {stego_tensor.shape}")
        
        # Convert to image
        stego_image = tensor_to_image(stego_tensor)
        logger.info(f"Stego image size: {stego_image.size}")
        
        # Convert to base64 for sending to frontend
        buffered = io.BytesIO()
        stego_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logger.info("Encoding completed successfully")
        
        return jsonify({'stego_image': img_str})
    except Exception as e:
        logger.error(f"Error in encode: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/decode', methods=['POST'])
def decode():
    try:
        logger.info("Received decode request")
        if 'image' not in request.files:
            logger.error("Missing image in request")
            return jsonify({'error': 'Missing image'}), 400
        
        # Get image
        image_file = request.files['image']
        image = Image.open(image_file)
        logger.info(f"Input image size: {image.size}")
        stego_tensor = image_to_tensor(image).to(device)
        logger.info(f"Stego tensor shape: {stego_tensor.shape}")
        logger.info(f"Stego tensor range: [{stego_tensor.min().item()}, {stego_tensor.max().item()}]")
        
        # Decode
        logger.info("Starting decoding process...")
        with torch.no_grad():
            recovered_tensor = decoder(stego_tensor)
        logger.info(f"Recovered tensor shape: {recovered_tensor.shape}")
        logger.info(f"Raw decoder output (min, max, mean): {recovered_tensor.min().item()}, {recovered_tensor.max().item()}, {recovered_tensor.mean().item()}")
        
        # Convert to text
        recovered_text = tensor_to_text(recovered_tensor)
        logger.info(f"Recovered text: {recovered_text}")
        
        if not recovered_text:
            logger.error("No text was recovered from the image")
            return jsonify({'error': 'Failed to recover text from image'}), 500
            
        logger.info("Decoding completed successfully")
        return jsonify({'text': recovered_text})
    except Exception as e:
        logger.error(f"Error in decode: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True) 