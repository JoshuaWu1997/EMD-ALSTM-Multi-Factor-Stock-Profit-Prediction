from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, CSVLogger

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0)
tbCallBack = TensorBoard(log_dir='G:/Tensorboard/_Graph', histogram_freq=1, write_graph=True, write_images=True)
early_stop = EarlyStopping(patience=30)
csv_log = CSVLogger('training.csv', append=True)
