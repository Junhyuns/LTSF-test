from .data_loader import Dataset_Custom

from torch.utils.data import DataLoader

def data_provider(args, flag):   
    if flag == "train":
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        # freq = args.freq
        
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        # freq = args.freq
    
    data_set = Dataset_Custom(config = args,
                              split_proportion = [0.7, 0.1, 0.2],
                              flag = flag)
    
    data_loader = DataLoader(data_set, 
                             batch_size = batch_size,
                             shuffle = shuffle_flag,
                             drop_last = drop_last,
                             num_workers = args.num_workers)
    
    return data_set, data_loader