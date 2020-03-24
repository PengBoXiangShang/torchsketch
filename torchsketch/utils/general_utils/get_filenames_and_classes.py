import os


def get_filenames_and_classes(dataset_dir):

    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            class_names.append(filename)

    
    class_names_to_ids = dict(zip(sorted(class_names), range(len(class_names))))

    return class_names_to_ids, len(class_names)