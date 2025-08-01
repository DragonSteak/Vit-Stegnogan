def save_model(encoder,decoder,critic,en_de_optimizer,cr_optimizer,metrics,ep):
    now = datetime.datetime.now()
    cover_score = metrics['val.cover_score'][-1]
    name = "%s_%s_%+.3f_%s.dat" % (encoder.name,decoder.name,cover_score,
                                   now.strftime("%Y-%m-%d_%H:%M:%S"))
    fname = os.path.join('.', 'myresults/model', name)
    states = {
            'state_dict_critic': critic.state_dict(),
            'state_dict_encoder': encoder.state_dict(),
            'state_dict_decoder': decoder.state_dict(),
            'en_de_optimizer': en_de_optimizer.state_dict(),
            'cr_optimizer': cr_optimizer.state_dict(),
            'metrics': metrics,
            'train_epoch': ep,
            'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
    }
    torch.save(states, fname)
    path='myresults/plots/train_%s_%s_%s'% (encoder.name,decoder.name,now.strftime("%Y-%m-%d_%H:%M:%S"))
    try:
      os.mkdir(os.path.join('.', path))
    except Exception as error:
      print(error)

    plot('encoder_mse', ep, metrics['val.encoder_mse'], path, True)
    plot('decoder_loss', ep, metrics['val.decoder_loss'], path, True)
    plot('decoder_acc', ep, metrics['val.decoder_acc'], path, True)
    plot('cover_score', ep, metrics['val.cover_score'], path, True)
    plot('generated_score', ep, metrics['val.generated_score'], path, True)
    plot('ssim', ep, metrics['val.ssim'], path, True)
    plot('psnr', ep, metrics['val.psnr'], path, True)
    plot('bpp', ep, metrics['val.bpp'], path, True)

def fit_gan(encoder,decoder,critic,en_de_optimizer,cr_optimizer,metrics,train_loader,valid_loader):
      for ep in range(epochs):
        print("Epoch %d" %(ep+1))
        for cover, _ in notebook.tqdm(train_loader):
            gc.collect()
            cover = cover.to(device)
            N, _, H, W = cover.size()
            # sampled from the discrete uniform distribution over 0 to 2
            payload = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)
            cover_score = torch.mean(critic.forward(cover))
            generated_score = torch.mean(critic.forward(generated))

            cr_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            cr_optimizer.step()

            for p in critic.parameters():
                p.data.clamp_(-0.1, 0.1)
            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

        for cover, _ in notebook.tqdm(train_loader):
            gc.collect()
            cover = cover.to(device)
            N, _, H, W = cover.size()
            # sampled from the discrete uniform distribution over 0 to 2
            payload = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)
            decoded = decoder.forward(generated)
            encoder_mse = mse_loss(generated, cover)
            decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
            decoder_acc = (decoded >= 0.0).eq(
                payload >= 0.5).sum().float() / payload.numel()
            generated_score = torch.mean(critic.forward(generated))

            en_de_optimizer.zero_grad()
            (100 * encoder_mse + decoder_loss +
             generated_score).backward()  # Why 100?
            en_de_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

        for cover, _ in notebook.tqdm(valid_loader):
            gc.collect()
            cover = cover.to(device)
            N, _, H, W = cover.size()
            # sampled from the discrete uniform distribution over 0 to 2
            payload = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)
            decoded = decoder.forward(generated)

            encoder_mse = mse_loss(generated, cover)
            decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
            decoder_acc = (decoded >= 0.0).eq(
                payload >= 0.5).sum().float() / payload.numel()
            generated_score = torch.mean(critic.forward(generated))
            cover_score = torch.mean(critic.forward(cover))

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(
                ssim(cover, generated).item())
            metrics['val.psnr'].append(
                10 * torch.log10(4 / encoder_mse).item())
            metrics['val.bpp'].append(
                data_depth * (2 * decoder_acc.item() - 1))
        print('encoder_mse: %.3f - decoder_loss: %.3f - decoder_acc: %.3f - cover_score: %.3f - generated_score: %.3f - ssim: %.3f - psnr: %.3f - bpp: %.3f'
          %(encoder_mse.item(),decoder_loss.item(),decoder_acc.item(),cover_score.item(),generated_score.item(), ssim(cover, generated).item(),10 * torch.log10(4 / encoder_mse).item(),data_depth * (2 * decoder_acc.item() - 1)))
      save_model(encoder,decoder,critic,en_de_optimizer,cr_optimizer,metrics,ep)

if __name__ == '__main__':
  for func in [
            lambda: os.mkdir(os.path.join('.', 'results')),
            lambda: os.mkdir(os.path.join('.', 'results/model')),
            lambda: os.mkdir(os.path.join('.', 'results/plots'))]:  # create directories
    try:
      func()
    except Exception as error:
      print(error)
      continue

  METRIC_FIELDS = [
        'val.encoder_mse',
        'val.decoder_loss',
        'val.decoder_acc',
        'val.cover_score',
        'val.generated_score',
        'val.ssim',
        'val.psnr',
        'val.bpp',
        'train.encoder_mse',
        'train.decoder_loss',
        'train.decoder_acc',
        'train.cover_score',
        'train.generated_score',
  ]

  print('image')
  data_dir = '/content/Deep-Steganography-using-Steganogan/dv2k_Dataset'
  mu = [.5, .5, .5]
  sigma = [.5, .5, .5]
  transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(
                                        360, pad_if_needed=True),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mu, sigma)])
  train_set = datasets.ImageFolder(os.path.join(
        data_dir, "Train/"), transform=transform)
  train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True)
  valid_set = datasets.ImageFolder(os.path.join(
        data_dir, "Train/val/"), transform=transform)
  valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=4, shuffle=False)

  encoder = DenseEncoder(data_depth, hidden_size).to(device)
  decoder = DenseDecoder(data_depth, hidden_size).to(device)
  critic = BasicCritic(hidden_size).to(device)
  cr_optimizer = Adam(critic.parameters(), lr=1e-4)
  en_de_optimizer = Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=1e-4)
  metrics = {field: list() for field in METRIC_FIELDS}

  if LOAD_MODEL:
    if torch.cuda.is_available():
      checkpoint = torch.load(PATH)
    else:
      checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)

    critic.load_state_dict(checkpoint['state_dict_critic'])
    encoder.load_state_dict(checkpoint['state_dict_encoder'])
    decoder.load_state_dict(checkpoint['state_dict_decoder'])
    en_de_optimizer.load_state_dict(checkpoint['en_de_optimizer'])
    cr_optimizer.load_state_dict(checkpoint['cr_optimizer'])
    metrics=checkpoint['metrics']
    ep=checkpoint['train_epoch']
    date=checkpoint['date']
    critic.train(mode=False)
    encoder.train(mode=False)
    decoder.train(mode=False)
    print('GAN loaded: ', ep)
    print(critic)
    print(encoder)
    print(decoder)
    print(en_de_optimizer)
    print(cr_optimizer)
    print(date)
  else:
    fit_gan(encoder,decoder,critic,en_de_optimizer,cr_optimizer,metrics,train_loader,valid_loader)


from collections import Counter
def make_payload(width, height, depth, text):
    """
    This takes a piece of text and encodes it into a bit vector. It then
    fills a matrix of size (width, height) with copies of the bit vector.
    """
    message = text_to_bits(text) + [0] * 32

    payload = message
    while len(payload) < width * height * depth:
        payload += message

    payload = payload[:width * height * depth]

    return torch.FloatTensor(payload).view(1, depth, height, width)

def make_message(image):
    #image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
    image = image.to(device)

    image = decoder(image).view(-1) > 0
    image=torch.tensor(image, dtype=torch.uint8)

    # split and decode messages
    candidates = Counter()
    bits = image.data.cpu().numpy().tolist()
    for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
      candidate = bytearray_to_text(bytearray(candidate))
      if candidate:
          #print(candidate)
          candidates[candidate] += 1

    # choose most common message
    if len(candidates) == 0:
      raise ValueError('Failed to find message.')

    candidate, count = candidates.most_common(1)[0]
    return candidate


# to see one image
cover,*rest = next(iter(valid_set))
_, H, W = cover.size()
cover = cover[None].to(device)
text = "how are you"
payload = make_payload(W, H, data_depth, text)
payload = payload.to(device)
#generated = encoder.forward(cover, payload)
generated = test(encoder,decoder,data_depth,epochs,cover,payload)
text_return = make_message(generated)
print(text_return)

from imageio import imread, imwrite
import os
import torch

epochs = 64
data_depth = 4
test_folder = "/content/Deep-Steganography-using-Steganogan/dv2k_Dataset/Train/val"
save_path = os.path.join(test_folder, f"{data_depth}_{epochs}")
os.makedirs(save_path, exist_ok=True)

for filename in os.listdir(test_folder):
    filepath = os.path.join(test_folder, filename)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) or not os.path.isfile(filepath):
        continue

    print(f"Processing: {filepath}")

    cover_im = imread(filepath, pilmode='RGB') / 127.5 - 1.0
    cover = torch.FloatTensor(cover_im).permute(2, 1, 0).unsqueeze(0)
    cover_size = cover.size()

    text = "how are you"
    payload = make_payload(cover_size[3], cover_size[2], data_depth, text)
    cover = cover.to(device)
    payload = payload.to(device)

    with torch.no_grad():
        generated = encoder.forward(cover, payload)[0].clamp(-1.0, 1.0)

    generated_np = (generated.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    save_name = os.path.join(save_path, f"{data_depth}_{epochs}_{filename}")
    imwrite(save_name, generated_np.astype('uint8'))


import imageio.v2 as imageio
import torch
import matplotlib.pyplot as plt
import os

steg_folder = "/content/Deep-Steganography-using-Steganogan/dv2k_Dataset/Train/val/4_64"
filename = "4_64_0001.png"
image = imageio.imread(os.path.join(steg_folder, filename), pilmode='RGB') / 127.5 - 1.0

# For display
plt.imshow((image + 1.0) / 2.0)

# For model input
image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)  # CHW
text_return = make_message(image_tensor)
print(text_return)

##Input to outut (both encode decode in one cell)
from imageio import imread, imwrite

cover_im = imread("/content/Deep-Steganography-using-Steganogan/dv2k_Dataset/Train/val/myval/0004.png", pilmode='RGB') / 127.5 - 1.0
plt.imshow(cover_im)
cover = torch.FloatTensor(cover_im).permute(2, 1, 0).unsqueeze(0)
cover_size = cover.size()
# _, _, height, width = cover.size()
text = "WWE is a wrestling sport"
payload = make_payload(cover_size[3], cover_size[2], data_depth, text)

cover = cover.to(device)
payload = payload.to(device)
generated = encoder.forward(cover, payload)
text_return = make_message(generated)
print(text_return)

!pip install onnx


pip install torch torchvision onnx onnxruntime


# --- Set parameters ---
data_depth = 4        # set to your value
hidden_size = 32      # set to your value
image_height = 360    # set to your value
image_width = 360     # set to your value
checkpoint_path = '/content/Deep-Steganography-using-Steganogan/Image_models/DenseEncoder_DenseDecoder_0.042_2020-07-23_02_08_27.dat'  # update this

# --- Load models ---
encoder = DenseEncoder(data_depth, hidden_size)
decoder = DenseDecoder(data_depth, hidden_size)
checkpoint = torch.load(checkpoint_path, map_location='cpu')
encoder.load_state_dict(checkpoint['state_dict_encoder'])
decoder.load_state_dict(checkpoint['state_dict_decoder'])
encoder.eval()
decoder.eval()

# --- Create dummy inputs ---
dummy_image = torch.randn(1, 3, image_height, image_width)
dummy_payload = torch.randint(0, 2, (1, data_depth, image_height, image_width)).float()

# --- Export to ONNX ---
import torch.onnx
torch.onnx.export(
    encoder,
    (dummy_image, dummy_payload),
    "encoder.onnx",
    input_names=['image', 'payload'],
    output_names=['stego_image'],
    opset_version=11
)
torch.onnx.export(
    decoder,
    dummy_image,
    "decoder.onnx",
    input_names=['stego_image'],
    output_names=['decoded_payload'],
    opset_version=11
)

# --- (Optional) Test ONNX models ---
import onnxruntime as ort
encoder_session = ort.InferenceSession("encoder.onnx")
encoder_output = encoder_session.run(
    None,
    {
        "image": dummy_image.numpy(),
        "payload": dummy_payload.numpy()
    }
)
print("Encoder output shape:", encoder_output[0].shape)
decoder_session = ort.InferenceSession("decoder.onnx")
decoder_output = decoder_session.run(
    None,
    {
        "stego_image": encoder_output[0]
    }
)
print("Decoder output shape:", decoder_output[0].shape)

import torch

# Load the .dat file
data = torch.load("/content/Deep-Steganography-using-Steganogan/Image_models/DenseEncoder_DenseDecoder_0.042_2020-07-23_02_08_27.dat", map_location='cpu')

# Inspect what's inside
print(type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())


import torch

checkpoint = torch.load("/content/Deep-Steganography-using-Steganogan/Image_models/DenseEncoder_DenseDecoder_0.042_2020-07-23_02_08_27.dat", map_location='cpu')


# Load the .dat checkpoint (you already have this)
checkpoint = torch.load("/content/Deep-Steganography-using-Steganogan/Image_models/DenseEncoder_DenseDecoder_0.042_2020-07-23_02_08_27.dat", map_location='cpu')

# Use your in-notebook model classes
encoder = DenseEncoder()
decoder = DenseDecoder()

# Load the saved weights
encoder.load_state_dict(checkpoint['state_dict_encoder'])
decoder.load_state_dict(checkpoint['state_dict_decoder'])

# Set models to eval mode
encoder.eval()
decoder.eval()

# Save the models as .pth files
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")

# Optional (if you're in Colab): trigger file download
from google.colab import files
files.download("encoder.pth")
files.download("decoder.pth")