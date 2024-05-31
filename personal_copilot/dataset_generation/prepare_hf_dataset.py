# coding=utf-8
# Copyright 2024 Sourab Mangrulkar. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from datasets import Dataset

DATAFOLDER = "../../../finetuning/representations/token/data"
HF_DATASET_NAME = "sebothetramp/sketch_token_small"

def create_hf_dataset():
    df = pd.read_parquet(DATAFOLDER + "/sketch.parquet")

    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(DATAFOLDER + "/sketch_hf")
    # dataset.push_to_hub(HF_DATASET_NAME, private=False)

if __name__ == "__main__":
    create_hf_dataset()
