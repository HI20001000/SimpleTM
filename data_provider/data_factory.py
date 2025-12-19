from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, _build_svmd_settings
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    svmd_settings = _build_svmd_settings(
        use_svmd=args.use_svmd,
        svmd_cache_dir=args.svmd_cache_dir,
        svmd_k=args.svmd_k,
        svmd_max_modes=args.svmd_max_modes,
        svmd_max_iter=args.svmd_max_iter,
        svmd_max_runtime=args.svmd_max_runtime,
        svmd_downsample=args.svmd_downsample,
        svmd_long_series_len=args.svmd_long_series_len,
        svmd_long_series_iter=args.svmd_long_series_iter,
    )

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        svmd_settings=svmd_settings,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
