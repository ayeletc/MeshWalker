import os
import time
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import rnn_model
import dataset
import utils
import params_setting
import attention_model
import evaluate_classification
import evaluate_segmentation

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')


def train_val(params):
    utils.next_iter_to_keep = 10000  # 5000
    print(utils.color.BOLD + utils.color.RED + 'params.logdir :::: ', params.logdir, utils.color.END)
    print(utils.color.BOLD + utils.color.RED, os.getpid(), utils.color.END)
    utils.backup_python_files_and_params(params)

    # Set up datasets for training and for test
    # -----------------------------------------
    train_datasets = []
    train_ds_iters = []
    max_train_size = 0
    for i in range(len(params.datasets2use['train'])):
        if params.net == 'Transformer':
            mode = 'classification'  # also for segmentation to take the correct dbs
        else:
            mode = params.network_task        # mode = params.network_tasks[i]
        this_train_dataset, n_trn_items = dataset.tf_mesh_dataset(params, params.datasets2use['train'][i],
                                                                  mode=mode, size_limit=params.train_dataset_size_limit,
                                                                  shuffle_size=100, min_max_faces2use=params.train_min_max_faces2use,
                                                                  min_dataset_size=128,
                                                                  data_augmentation=params.train_data_augmentation)
        print('Train Dataset size:', n_trn_items)
        train_ds_iters.append(iter(this_train_dataset.repeat()))
        train_datasets.append(this_train_dataset)
        max_train_size = max(max_train_size, n_trn_items)
    train_epoch_size = max(8, int(max_train_size / params.n_walks_per_model / params.batch_size))
    #train_epoch_size = 2
    print('train_epoch_size:', train_epoch_size)
    if params.datasets2use['test'] is None:
        test_dataset = None
        n_tst_items = 0
    else: # TODO: change
        # test_dataset, n_tst_items = dataset.tf_mesh_dataset(params, params.datasets2use['test'][0],
        #                                                     mode=mode, size_limit=params.test_dataset_size_limit,
        #                                                     shuffle_size=100, min_max_faces2use=params.test_min_max_faces2use)
        test_dataset, n_tst_items = dataset.tf_mesh_dataset(params, params.datasets2use['train'][0],
                                                        mode=mode, size_limit=params.train_dataset_size_limit,
                                                        shuffle_size=100,
                                                        min_max_faces2use=params.train_min_max_faces2use,
                                                        min_dataset_size=128,
                                                        data_augmentation=params.train_data_augmentation)
    print(' Test Dataset size:', n_tst_items)

    # Set up RNN model and optimizer
    # ------------------------------
    if params.net_start_from_prev_net is not None:
        init_net_using = params.net_start_from_prev_net
    else:
        init_net_using = None

    if params.optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate[0], clipnorm=params.gradient_clip_th)
    elif params.optimizer_type == 'cycle':
        @tf.function
        def _scale_fn(x):
            x_th = 500e3 / params.cycle_opt_prms.step_size
            if x < x_th:
                return 1.0
            else:
                return 0.5
        lr_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params.cycle_opt_prms.initial_learning_rate,
                                                        maximal_learning_rate=params.cycle_opt_prms.maximal_learning_rate,
                                                        step_size=params.cycle_opt_prms.step_size,
                                                        scale_fn=_scale_fn, scale_mode="cycle", name="MyCyclicScheduler")
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=params.gradient_clip_th)
    elif params.optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=params.learning_rate[0], decay=0, momentum=0.9, nesterov=True, clipnorm=params.gradient_clip_th)
    else:
        raise Exception('optimizer_type not supported: ' + params.optimizer_type)

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    if params.net == 'RnnWalkNet':
        dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, init_net_using, optimizer=optimizer)
    elif params.net == 'Transformer':
        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8
        dropout_rate = 0.1
        dnn_model = attention_model.Transformer(num_layers=num_layers,
                                                d_model=d_model,
                                                num_heads=num_heads,
                                                dff=dff,
                                                #input_vocab_size=tokenizers.pt.get_vocab_size(),
                                                #target_vocab_size=params.n_classes,
                                                pe_input=1000,  # TODO: enlarge?
                                                pe_target=1000,
                                                params=params,
                                                rate=dropout_rate)

    # Other initializations
    # ---------------------
    time_msrs = {}
    time_msrs_names = ['train_step', 'get_train_data', 'test']
    for name in time_msrs_names:
        time_msrs[name] = 0
    seg_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='seg_train_accuracy')

    train_log_names = ['seg_loss']
    train_logs = {name: tf.keras.metrics.Mean(name=name) for name in train_log_names}
    train_logs['seg_train_accuracy'] = seg_train_accuracy

    # Train / test functions
    # ----------------------
    if params.last_layer_actication is None:
        if params.network_task == 'semantic_segmentation':
            seg_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            seg_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                     sample_weight=tf.constant([1/32., 31/32.]))
    else:
        seg_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # @tf.function
    def train_step(model_ftrs_, labels_, one_label_per_model):
        sp = model_ftrs_.shape
        model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
        with tf.GradientTape() as tape:
            if one_label_per_model:
                labels = tf.reshape(tf.transpose(tf.stack((labels_,)*params.n_walks_per_model)), (-1,))
                if params.net == 'RnnWalkNet':
                    predictions = dnn_model(model_ftrs)
                elif params.net == 'Transformer':
                    # prev try
                    # predictions, attenetion_weights = dnn_model(model_ftrs, labels[:, tf.newaxis],
                    #                                             True, None, None, None)

                    # classify with "start" value = params.n_classes
                    model_ftrs = tf.concat([tf.zeros([sp[0], 1, sp[2]]), model_ftrs], axis=1)
                    labels = expand_labels_dim(labels, sp[0], params.output_size, params.n_classes)
                    enc_padding_mask, combined_mask, dec_padding_mask = attention_model.create_masks(model_ftrs,
                                                                                                     labels)
                    predictions, attenetion_weights = dnn_model(model_ftrs, labels, True,
                                                                enc_padding_mask=None,
                                                                look_ahead_mask=combined_mask,
                                                                dec_padding_mask=None)
            else:
                skip = params.min_seq_len  # do not classify all the vertices, only (seq_len-skip)
                if params.net == 'RnnWalkNet':
                    labels = tf.reshape(labels_, (-1, sp[-2]))
                    predictions = dnn_model(model_ftrs)[:, skip:]
                    labels = labels[:, skip + 1:]
                elif params.net == 'Transformer':
                    model_ftrs = tf.concat([tf.zeros([sp[0], 1, sp[2]]), model_ftrs], axis=1)
                    labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
                    # labels = tf.cast(tf.repeat(labels[:, tf.newaxis], repeats=params.seq_len, axis=1), tf.int32)
                    # labels = tf.concat([tf.cast(tf.ones([sp[0], 1])*params.n_classes, tf.int32), labels], axis=1)
                    labels = expand_labels_dim(labels, sp[0], params.output_size, params.n_classes)
                    enc_padding_mask, combined_mask, dec_padding_mask = attention_model.create_masks(model_ftrs,
                                                                                labels)
                    predictions, attenetion_weights = dnn_model(model_ftrs, labels,
                                                                True, enc_padding_mask=None, look_ahead_mask=combined_mask,
                                                                dec_padding_mask=None)

            def loss_function(real, pred):
                # mask = tf.math.logical_not(tf.math.equal(real, 0))
                loss_ = seg_loss(real[:, 1:], pred[:, 1:])*30/31
                loss_ += seg_loss(real[:, 0], pred[:, 0])*1/31
                # mask = tf.cast(mask, dtype=loss_.dtype)
                # loss_ *= mask
                # return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
                return tf.reduce_sum(loss_)

            seg_train_accuracy(labels, predictions)
            loss = loss_function(labels, predictions)

        gradients = tape.gradient(loss, dnn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))

        train_logs['seg_loss'](loss)

        return loss

    # @tf.function  # keep in comment
    def test_step(model_ftrs_, labels_, name, epoch, test_iter, one_label_per_model):
        # make attention (log) directory
        if not os.path.isdir(params.attention_dir_name):
            os.mkdir(params.attention_dir_name)
        cur_attention_dir = os.path.join(params.attention_dir_name, str(epoch))
        if not os.path.isdir(cur_attention_dir):
            os.mkdir(cur_attention_dir)
        sp = model_ftrs_.shape
        model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))

        if one_label_per_model:
            labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
            if params.net == 'RnnWalkNet':
                predictions = dnn_model(model_ftrs)
                best_pred = tf.math.argmax(predictions, axis=-1)
            elif params.net == 'Transformer':
                predictions, predictions_prob, attention = evaluate_classification.evaluate(dnn_model,
                                                                                            model_ftrs, params)
                best_pred = predictions[:, -1]
        else:
            if params.net == 'RnnWalkNet':
                labels = tf.reshape(labels_, (-1, sp[-2]))
                skip = params.min_seq_len
                predictions = dnn_model(model_ftrs)
                labels = labels[:, skip + 1:]

            elif params.net == 'Transformer':
                labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
                labels = expand_labels_dim(labels, sp[0], params.output_size, params.n_classes) # repeats: params.output_size
                # best_pred, predictions_prob, attention = evaluate_segmentation.evaluate(dnn_model,
                #                                                                             model_ftrs, params)
                best_pred, predictions_prob = evaluate_segmentation.evaluate(dnn_model, model_ftrs, params,
                                                                 skip=0, get_attention=False)

        if params.net == 'RnnWalkNet':
            best_pred = tf.math.argmax(predictions, axis=-1)
            test_accuracy(labels, predictions)
            confusion = tf.math.confusion_matrix(labels=tf.reshape(labels, (-1,)),
                                                 predictions=tf.reshape(best_pred, (-1,)),
                                                 num_classes=params.n_classes)

        elif params.net == 'Transformer':
            # np.sum((tf.math.equal(labels, tf.cast(best_pred, tf.int32)))) / labels.shape[0]
            test_accuracy.update_state(labels[:, 1:], predictions_prob)
            confusion = tf.math.confusion_matrix(labels=tf.reshape(labels, (-1,)),
                                                 predictions=tf.reshape(best_pred, (-1,)),
                                                num_classes=params.n_classes+1)
        # if params.net == 'Transformer':
        #     utils.save_attention(name, model_ftrs, attention, vetrices_indices, cur_attention_dir, test_iter)
        return confusion
    # -------------------------------------

    # Loop over training EPOCHs
    # -------------------------
    jj = 0
    one_label_per_model = params.network_task == 'classification'
    next_iter_to_log = 0
    e_time = 0
    accrcy_smoothed = tb_epoch = last_loss = None
    all_confusion = {}
    with tf.summary.create_file_writer(params.logdir).as_default():
        epoch = 0
        while optimizer.iterations.numpy() < params.iters_to_train + train_epoch_size * 2:
            epoch += 1
            str_to_print = str(os.getpid()) + ') Epoch' + str(epoch) + ', iter ' + str(optimizer.iterations.numpy())

            # Save some logs & infos
            utils.save_model_if_needed(optimizer.iterations, dnn_model, params)
            if tb_epoch is not None:
                e_time = time.time() - tb_epoch
                tf.summary.scalar('time/one_epoch', e_time, step=optimizer.iterations)
                tf.summary.scalar('time/av_one_trn_itr', e_time / n_iters, step=optimizer.iterations)
                for name in time_msrs_names:
                    if time_msrs[name]:  # if there is something to save
                        tf.summary.scalar('time/' + name, time_msrs[name], step=optimizer.iterations)
                        time_msrs[name] = 0
            tb_epoch = time.time()
            n_iters = 0
            tf.summary.scalar(name="train/learning_rate", data=optimizer._decayed_lr(tf.float32), step=optimizer.iterations)
            tf.summary.scalar(name="mem/free", data=utils.check_mem_and_exit_if_full(), step=optimizer.iterations)
            gpu_tmpr = utils.get_gpu_temprature()
            if gpu_tmpr > 95:
                print('GPU temprature is too high!!!!!')
                exit(0)
            tf.summary.scalar(name="mem/gpu_tmpr", data=gpu_tmpr, step=optimizer.iterations)

            # Train one EPOC
            jj += 1
            train_logs['seg_loss'].reset_states()
            tb = time.time()
            for iter_db in range(train_epoch_size):
                for dataset_id in range(len(train_datasets)):
                    name, model_ftrs_, labels = train_ds_iters[dataset_id].next()
                    sp = model_ftrs_.shape
                    model_ftrs_ = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
                    model_ftrs = tf.cast(model_ftrs_[:, :, :3], tf.float32)
                    vetrices_indices = tf.cast(model_ftrs_[0, :, 3], tf.int16)
                    dataset_type = utils.get_dataset_type_from_name(name)
                    time_msrs['get_train_data'] += time.time() - tb
                    n_iters += 1
                    tb = time.time()
                    if params.train_loss[dataset_id] == 'cros_entr':
                        train_step(model_ftrs, labels, one_label_per_model=one_label_per_model)#TBD
                        loss2show = 'seg_loss'
                    else:
                        raise Exception('Unsupported loss_type: ' + params.train_loss[dataset_id])
                    time_msrs['train_step'] += time.time() - tb
                    tb = time.time()
                if iter_db == train_epoch_size - 1:
                    str_to_print += ', TrnLoss: ' + str(round(train_logs[loss2show].result().numpy(), 2))

            # Dump training info to tensorboard
            if optimizer.iterations >= next_iter_to_log:
                for k, v in train_logs.items():
                    if v.count.numpy() > 0:
                        tf.summary.scalar('train/' + k, v.result(), step=optimizer.iterations)
                        v.reset_states()
                next_iter_to_log += params.log_freq

            # Run test on part of the test set
            if test_dataset is not None and jj % 23 == 0:
                n_test_iters = 0
                for name, model_ftrs_, labels in test_dataset:
                    sp = model_ftrs_.shape
                    model_ftrs_ = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
                    model_ftrs = tf.cast(model_ftrs_[:, :, :3], tf.float32)
                    vetrices_indices = tf.cast(model_ftrs_[:, :, 3], tf.int16)
                    n_test_iters += model_ftrs.shape[0]
                    if n_test_iters > params.n_models_per_test_epoch:
                        break
                    confusion = test_step(model_ftrs, labels, name, epoch, n_test_iters,
                                          one_label_per_model=one_label_per_model)
                    dataset_type = utils.get_dataset_type_from_name(name)
                    if dataset_type in all_confusion.keys():
                        all_confusion[dataset_type] += confusion
                    else:
                        all_confusion[dataset_type] = confusion
                # Dump test info to tensorboard
                if accrcy_smoothed is None:
                    accrcy_smoothed = test_accuracy.result()
                accrcy_smoothed = accrcy_smoothed * .9 + test_accuracy.result() * 0.1
                tf.summary.scalar('test/accuracy_' + dataset_type, test_accuracy.result(), step=optimizer.iterations)
                str_to_print += ', test/accuracy_' + dataset_type + ': ' + str(round(test_accuracy.result().numpy(), 2))
                test_accuracy.reset_states()
                time_msrs['test'] += time.time() - tb

            str_to_print += ', time: ' + str(round(e_time, 1))
            print(str_to_print)

    return last_loss


def expand_labels_dim(labels, batch_size, output_size, n_classes):
    labels = tf.cast(tf.repeat(labels[:, tf.newaxis], repeats=output_size-1, axis=1), tf.int32)
    return tf.concat([tf.cast(tf.ones([batch_size, 1]) * n_classes, tf.int32), labels], axis=1)


def get_params(job, job_part):
    # Classifications
    job = job.lower()
    if job == 'modelnet40' or job == 'modelnet':
        params = params_setting.modelnet_params()

    if job == 'shrec11':
        params = params_setting.shrec11_params(job_part)

    if job == 'cubes':
        params = params_setting.cubes_params()

    # Semantic Segmentations
    if job == 'human_seg':
        params = params_setting.human_seg_params()

    if job == 'coseg':
        params = params_setting.coseg_params(job_part)   #  job_part can be : 'aliens' or 'chairs' or 'vases'

    return params


def run_one_job(job, job_part):
    params = get_params(job, job_part)

    train_val(params)


def get_all_jobs():
    jobs = [
      'shrec11', 'shrec11', 'shrec11',
      'shrec11', 'shrec11', 'shrec11',
      'coseg', 'coseg', 'coseg',
      'human_seg',
      'cubes',
      'modelnet40',
    ][6:]
    job_parts = [
      '10-10_A', '10-10_B', '10-10_C',
      '16-04_A', '16-04_B', '16-04_C',
      'aliens', 'vases', 'chairs',
      None,
      None,
      None,
    ][6:]

    return jobs, job_parts


if __name__ == '__main__':
    np.random.seed(0)
    utils.config_gpu()

    if len(sys.argv) <= 1:
        print('Use: python train_val.py <job> <part>')
        print('<job> can be one of the following: shrec11 / coseg / human_seg / cubes / modelnet40')
        print('<job> can be also "all" to run all of the above.')
        print('<part> should be used in case of shrec11 or coseg datasets.')
        print('For shrec11 it should be one of the follows: 10-10_A / 10-10_B / 10-10_C / 16-04_A / 16-04_B / 16-04_C')
        print('For coseg it should be one of the follows: aliens / vases / chairs')
        print('For example: python train_val.py shrec11 10-10_A')
    else:
        job = sys.argv[1]
        job_part = sys.argv[2] if len(sys.argv) > 2 else '-'

        if job.lower() == 'all':
            jobs, job_parts = get_all_jobs()
            for job_, job_part in zip(jobs, job_parts):
                run_one_job(job_, job_part)
        else:
            run_one_job(job, job_part)
