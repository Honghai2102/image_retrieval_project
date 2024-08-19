import cv2
import os
import chromadb
from tqdm import tqdm
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import pickle
import time


def save_files(files_path, test_files_path):
    dirs = 'Image_Retrieval/vector_db'
    os.makedirs(dirs, exist_ok=True)
    with open(os.path.join(dirs, 'files_path.pkl'), 'wb') as f:
        pickle.dump(files_path, f)
    with open(os.path.join(dirs, 'test_files_path.pkl'), 'wb') as f:
        pickle.dump(test_files_path, f)


def create_embedding_ids(files_path):
    ids = []
    embeddings = []
    for id_file_path, file_path in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_file_path}')
        image = cv2.imread(file_path)
        embeddings.append(EMBEDDING_FUNCTION._encode_image(image))
    return embeddings, ids


def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = path + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + "/" + filename
            files_path.append(filepath)
    return files_path


def main():
    files_path = get_files_path(path='Image_Retrieval/data/train')
    test_files_path = get_files_path(path='Image_Retrieval/data/test')
    embeddings, ids = create_embedding_ids(files_path)

    name = "my_collection"
    metadata = {"hnsw:space": "l2"}
    my_collection = CHROMA_CLIENT.create_collection(
        name=name,
        metadata=metadata
    )
    my_collection.add(embeddings=embeddings, ids=ids)

    save_files(files_path, test_files_path)


if __name__ == "__main__":
    start_time = time.time()
    CLASS_NAME = sorted(list(os.listdir('Image_Retrieval/data/train')))
    EMBEDDING_FUNCTION = OpenCLIPEmbeddingFunction()
    CHROMA_CLIENT = chromadb.PersistentClient(path="Image_Retrieval/vector_db")
    main()
    print(f"Runtime: {(time.time()-start_time):.5f}s\n")
