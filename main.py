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