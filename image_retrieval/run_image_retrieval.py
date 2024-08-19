import os
import chromadb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import pickle
import time


def plot_results(image_path, files_path, results):
    query_image = Image.open(image_path).resize((448, 448))
    images = [query_image]
    class_name = []
    for id_img in results['ids'][0]:
        id_img = int(id_img.split('_')[-1])
        img_path = files_path[id_img]
        img = Image.open(img_path).resize((448, 448))
        images.append(img)
        class_name.append(img_path.split('/')[2])

    _, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if i == 0:
            ax.set_title(f"Query Image: {image_path.split('/')[2]}")
        else:
            ax.set_title(f"Top {i+1}: {class_name[i-1]}")
        ax.axis('off')
    plt.show()


def search(image_path, collection, n_results=5):
    query_image = Image.open(image_path)
    results = collection.query(
        query_embeddings=[EMBEDDING_FUNCTION._encode_image(
            image=np.array(query_image))],
        n_results=n_results
    )
    return results


def load_data(persist_directory):
    with open(os.path.join(persist_directory, 'files_path.pkl'), 'rb') as f:
        files_path = pickle.load(f)
    with open(os.path.join(persist_directory, 'test_files_path.pkl'), 'rb') as f:
        test_files_path = pickle.load(f)
    return files_path, test_files_path


def main():
    files_path, test_files_path = load_data('Image_Retrieval/vector_db')
    test_path = test_files_path[55]

    my_collection = CHROMA_CLIENT.get_collection(name="my_collection")

    results = search(image_path=test_path,
                     collection=my_collection)

    plot_results(image_path=test_path,
                 files_path=files_path, results=results)


if __name__ == "__main__":
    start_time = time.time()
    EMBEDDING_FUNCTION = OpenCLIPEmbeddingFunction()
    CHROMA_CLIENT = chromadb.PersistentClient(path="Image_Retrieval/vector_db")
    main()
    print(f"Runtime: {(time.time()-start_time):.5f}s\n")
