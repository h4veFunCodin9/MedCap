import torch
import random
from data.language import EOS_INDEX, SOS_INDEX
import time
from .evaluate import evaluate_pairs, display_randomly
from .utils import time_since, show_plot
import numpy as np
from model.utils import save_model


# forward one sample
def train(input_variables, seg_target_variables, cap_target_variables, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, config):
    encoder_optimizer.zero_grad()
    if not config.OnlySeg:
        decoder_optimizer.zero_grad()

    seg_loss, cap_loss = 0, 0

    total_word_num = 0

    # input_variable: 1 x 3 x 512 x 512, cap_target_variable: num_sent x len_sent, stop_target_variable: max_num_sent x 1
    def _one_pass(input_variable, seg_target_variable, cap_target_variable, only_seg):

        _im_embedding, pred_seg = encoder(input_variable)  # [1, HiddenSize]

        # compute segmentation loss
        _seg_loss = criterion(pred_seg, seg_target_variable)

        if only_seg:
            return _seg_loss, 0, 0, 0, 0

        _word_num = 0

        _cap_loss = 0

        teacher_forcing = True if random.random() < 0.5 else False
        sent_target_variable = cap_target_variable

        _sent_len = (sent_target_variable == EOS_INDEX).nonzero().data[0][0] + 1
        _word_seen_num = 0

        decoder_hidden = _im_embedding  # init RNN using topic vector
        decoder_input = torch.autograd.Variable(torch.LongTensor([[SOS_INDEX, ]]))
        decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

        if teacher_forcing:
            for word_i in range(_sent_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                _cap_loss += criterion(decoder_output[0], sent_target_variable[word_i])
                decoder_input = sent_target_variable[word_i]
                _word_seen_num += 1
        else:
            for word_i in range(_sent_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                _cap_loss += criterion(decoder_output[0], sent_target_variable[word_i])
                topv, topi = decoder_output[0].data.topk(1)
                ni = topi[0][0]

                decoder_input = torch.autograd.Variable(torch.LongTensor([[ni, ]]))
                decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

                _word_seen_num += 1
                if ni == EOS_INDEX:
                    break
            _word_num += _word_seen_num

        return _seg_loss, _cap_loss, _word_num

    batch_size = len(input_variables)
    for i in range(batch_size):
        cap_target_variable = cap_target_variables[i]
        seg_target_variable = seg_target_variables[i]
        input_variable = input_variables[i]

        cur_seg_loss, cur_cap_loss, cur_word_num = _one_pass(input_variable, seg_target_variable,
                                                             cap_target_variable, config.OnlySeg)

        total_word_num += cur_word_num

        cap_loss += cur_cap_loss
        seg_loss += cur_seg_loss

    if config.OnlySeg:
        loss = seg_loss
    else:
        loss = config.CapLoss_Weight * cap_loss + config.SegLoss_Weight * seg_loss
    loss.backward()

    encoder_optimizer.step()
    if not config.OnlySeg:
        decoder_optimizer.step()
        return cap_loss.data[0] / total_word_num, seg_loss.data[0] / batch_size
    else:
        return 0, seg_loss.data[0] / batch_size


# for train multiple iterations
def train_iters(model, train_dataset, val_dataset, config, start_iter=1):

    encoder = model['encoder']
    decoder = model['decoder']

    n_iters = config.NumIters
    batch_size = config.BatchSize
    print_every = config.PrintFrequency
    plot_every = config.PlotFrequency

    encoder_optimizer = torch.optim.SGD(params=encoder.parameters(), lr=config.LR, momentum=config.Momentum)
    decoder_optimizer = None
    if not config.OnlySeg:
        decoder_optimizer = torch.optim.SGD(params=decoder.parameters(), lr=config.LR, momentum=config.Momentum)

    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()

    print_seg_loss_total = 0
    print_caption_loss_total = 0
    print_loss_total = 0

    plot_seg_loss_total = 0
    plot_caption_loss_total = 0
    plot_loss_total = 0

    plot_losses = []
    plot_seg_losses = []
    plot_caption_losses = []

    for iter in range(int(start_iter), n_iters+int(start_iter)):
        train_dataset.shuffle()
        dataset_index, batch_index, dataset_size = 0, 0, len(train_dataset)
        while dataset_index + batch_size < dataset_size:
            current_pairs = [train_dataset[dataset_index+i] for i in range(batch_size)]

            dataset_index += batch_size
            batch_index += 1

            input_variables = [p[0] for p in current_pairs]
            seg_target_variables = [p[1] for p in current_pairs]
            cap_target_variables = [p[2] for p in current_pairs]

            caption_loss, seg_loss = train(input_variables, seg_target_variables, cap_target_variables, encoder,
                                            decoder, encoder_optimizer, decoder_optimizer, criterion, config)
            # dynamic adjust the learning rate
            # encoder_scheduler.step(seg_loss)

            loss = caption_loss + seg_loss
            # if not config.OnlySeg:
            #    sent_decoder_scheduler.step(loss)
            #    word_decoder_scheduler.step(loss)

            print_loss_total += loss
            print_seg_loss_total += seg_loss
            print_caption_loss_total += caption_loss

            plot_seg_loss_total += seg_loss
            plot_caption_loss_total += caption_loss
            plot_loss_total += loss

            if dataset_index % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_seg_loss_avg = print_seg_loss_total / print_every
                print_caption_loss_avg = print_caption_loss_total / print_every

                if config.OnlySeg:
                    iou, _ = evaluate_pairs(model, train_dataset.lang, val_dataset, config)
                    print(
                        '[Iter: %d, Batch: %d]%s (%d %d%%) loss = %.3f, seg_loss = %.3f, (val) '
                        'iou = [%.3f, %.3f, %.3f, %.3f]' %
                        (iter, batch_index, time_since(start, dataset_index / dataset_size), dataset_index,
                         dataset_index / dataset_size * 100,
                         print_loss_avg, print_seg_loss_avg, iou[0], iou[1], iou[2], iou[3]))
                else:
                    iou, bleu_scores = evaluate_pairs(model, train_dataset.lang, val_dataset, config)
                    print('[Iter: %d, Batch: %d]%s (%d %d%%) loss = %.3f, seg_loss = %.3f, '
                          'caption_loss = %.3f; Val: iou = [%.3f, %.3f, %.3f, %.3f], '
                          'bleu_score = [%.3f, %.3f, %.3f, %.3f]' %
                          (iter, batch_index, time_since(start, dataset_index / dataset_size), dataset_index,
                           dataset_index / dataset_size * 100, print_loss_avg, print_seg_loss_avg,
                           print_caption_loss_avg, iou[0], iou[1], iou[2], iou[3],
                           bleu_scores[0], bleu_scores[1], bleu_scores[2], bleu_scores[3]))

                print_loss_total, print_seg_loss_total, print_caption_loss_total = 0, 0, 0

                if not config.OnlySeg:
                    display_randomly(model, train_dataset.lang, val_dataset, config)

            if batch_index % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_seg_loss_avg = plot_seg_loss_total / plot_every
                plot_caption_loss_avg = plot_caption_loss_total / plot_every

                plot_losses.append(plot_loss_avg)
                plot_seg_losses.append(plot_seg_loss_avg)
                plot_caption_losses.append(plot_caption_loss_avg)

                plot_loss_total = 0
                plot_seg_loss_total = 0
                plot_caption_loss_total = 0

        if not config.OnlySeg:
            val_iou, val_bleu_scores = evaluate_pairs(model, train_dataset.lang, val_dataset, config)
            print("[Iter {}] Validation IOU: {:.3f} {:.3f} {:.3f} {:.3f}; BLEU: {:.3f} {:.3f} {:.3f} {:.3f}".format(
                iter, val_iou[0], val_iou[1], val_iou[2], val_iou[3], val_bleu_scores[0], val_bleu_scores[1],
                val_bleu_scores[2], val_bleu_scores[3]))

        if iter % config.SaveFrequency == 0:
            save_model(model, config, suffix='_'+str(iter))

    show_plot(plot_losses, config.StoreRoot, name="loss")
    show_plot(plot_seg_losses, config.StoreRoot, name='seg_loss')
