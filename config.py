epochs = 64
data_depth = 4
hidden_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL=True
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_0.041_2020-07-25_15:31:19.dat'
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_-0.003_2020-07-24_20:01:33.dat'
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_-0.022_2020-07-24_05:11:17.dat'
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_-0.041_2020-07-23_23:01:25.dat'
PATH='/content/Deep-Steganography-using-Steganogan/Image_models/DenseEncoder_DenseDecoder_0.042_2020-07-23_02_08_27.dat' ##Depth4Epoch64
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_0.005_2020-07-22_20:05:49.dat'
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_-0.019_2020-07-22_15:02:29.dat'
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_-0.020_2020-07-22_13:43:02.dat'
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_+0.048_2020-07-22_12:21:23.dat'
#PATH='/content/drive/My Drive/myresults/model/DenseEncoder_DenseDecoder_+0.017_2020-07-22_08:18:00.dat'

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