import math
from multiprocessing import Process





def _split_urls_4_subprocessing(processing_nums, urls):

    url_list_4_subprocessing = []
    n_sample = len(urls)
    num_per_subprocessing = int(math.ceil(n_sample / processing_nums))
    for i in range(processing_nums):
        start_ndx = i * num_per_subprocessing
        end_ndx = min((i + 1) * num_per_subprocessing, n_sample)
        url_list_4_subprocessing.append(urls[start_ndx : end_ndx])
    
    return url_list_4_subprocessing
        



def multiprocessing_acceleration(target_function, url_list, processing_nums = 8, *args):
    
    url_list_4_subprocessing = _split_urls_4_subprocessing(processing_nums, url_list)
    assert len(url_list_4_subprocessing) == processing_nums
    

    process = list()
      

    for i in range(processing_nums):

        parameter_list = url_list_4_subprocessing[i],
        parameter_list += args

        process.append(Process(target = target_function, args = parameter_list))

        del parameter_list
    
    for p in process:
        p.start()
    
    for p in process:
        p.join()



    print('All subprocesses done.')

        


    