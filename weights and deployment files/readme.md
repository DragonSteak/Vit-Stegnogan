For creating deployment files use the model weights to train and test the model and then import into pytorch script or onnx files.
its easier to use the encoder and decoder pth individually for easier lightweight deploying of just encoder and reusablity of decoder. 
Use for the SGAN setup
| File Name                         | Format             | Purpose                         |
| --------------------------------- | ------------------ | ------------------------------- |
| `encoder.pth`, `decoder.pth`      | PyTorch checkpoint | Encoder/Decoder weights         |
| `encoder.onnx`, `decoder.onnx`    | ONNX               | Exported for deployment         |
| `DenseEncoder_DenseDecoder_*.dat` | Binary             | Custom checkpoint with metadata |
