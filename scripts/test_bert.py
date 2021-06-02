import pandas as pd
import numpy as np
from bert_serving.client import BertClient

if __name__ == '__main__':
    dataset = pd.read_csv('data/pc_specs.csv', index_col=0)
    list_to_encode = dataset["CPU"]
    print(list(list_to_encode.unique()))

    tech_specs = dataset.drop(["Price (€)", "Comments (don't use commas)"], axis=1).values.tolist()
    tech_specs_list = [' '.join(map(str, i)) for i in tech_specs]

    bc = BertClient(
        check_length=False,
        port=5555,
        port_out=5556,
        output_fmt='ndarray',
        timeout=3000
    )
    embeddings = bc.encode(tech_specs_list)
    print(embeddings.shape)

    # save_data = np.concatenate((np.arange(embeddings.shape[0])[:,None], embeddings), axis=1)
    np.savetxt("data/tech_specs.csv", embeddings, delimiter='\t')
