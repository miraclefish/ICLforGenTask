from datasets import load_dataset, load_from_disk
from openicl import DatasetReader
from openicl import TopkRetriever
from openicl import PromptTemplate, PPLInferencer
from openicl import AccEvaluator
from accelerate import Accelerator
import argparse
import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

SBERT_CODE = {'all-mpnet-base-v2': 'AMB2',
              'paraphrase-mpnet-base-v2': 'PMB2'}

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset_name', type=str, default='SST-2')
    args.add_argument('--model_name', type=str, default='gpt2-large')
    args.add_argument('--sentence_transformers_name', type=str, default='all-mpnet-base-v2')
    args.add_argument('--ice_num', type=int, default=3)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--dataset_root_path', type=str, default='/apdcephfs_cq8/share_2992827/public/yifeiyu/Datasets')
    args.add_argument('--model_root_path', type=str, default='/apdcephfs_cq8/share_2992827/public/yifeiyu/LLM')
    args.add_argument('--output_json_filepath', type=str, default='./icl_inference_output')
    args.add_argument('--retriever_type', type=str, default='Topk')
    args.add_argument('--ascending_order', action='store_true')
    args.add_argument('--ddp', action='store_true')
    args.add_argument('--debug', action='store_false')
    args = args.parse_args()
    return args

def main(args):

     # Accelerate Prepare
     if args.ddp:
          accelerator = Accelerator()
     else:
          accelerator = None

     output_json_filepath = f"{args.output_json_filepath}/{args.retriever_type}"
     output_json_filename = f"{args.dataset_name}_{args.ice_num}_{args.model_name}_{SBERT_CODE[args.sentence_transformers_name]}"

     dataset_path = os.path.join(args.dataset_root_path, f"UDR_{args.dataset_name}")
     model_path = os.path.join(args.model_root_path, args.model_name)
     sentence_transformers_name = os.path.join(args.model_root_path, args.sentence_transformers_name)

     # Loading dataset from huggingface 
     dataset = load_from_disk(dataset_path)
     data = DatasetReader(dataset, input_columns=['sentence'], output_column='label',
                          ds_size=64 if args.debug else None)
     # template = PromptTemplate(template={
     #                                         1: '</E>Positive Movie Review: </text>',
     #                                         0: '</E>Negative Movie Review: </text>' 
     #                                    },
     #                          column_token_map={'sentence' : '</text>'},
     #                          ice_token='</E>'
     #           )
     
     # template = PromptTemplate(template={
     #                                         1: '</E></text>\nIt was great .\n\n',
     #                                         0: '</E></text>\nIt was terrible .\n\n' 
     #                                    },
     #                          column_token_map={'sentence' : '</text>'},
     #                          ice_token='</E>'
     #           )
     
     template = PromptTemplate(template={
                                             0: '</E></text>\nIt was terrible .\n\n',
                                             1: '</E></text>\nIt was bad .\n\n',
                                             2: '</E></text>\nIt was okay .\n\n',
                                             3: '</E></text>\nIt was good .\n\n',
                                             4: '</E></text>\nIt was great .\n\n'
                                        },
                              column_token_map={'sentence' : '</text>'},
                              ice_token='</E>'
               )

     retriever = TopkRetriever(data, ice_num=args.ice_num, index_split='train', test_split='test',
                               batch_size=args.batch_size, tokenizer_name=sentence_transformers_name,
                               sentence_transformers_model_name=sentence_transformers_name,
                               ascending_order=args.ascending_order,
                               accelerator=accelerator)

     inferencer = PPLInferencer(model_name=model_path, batch_size=args.batch_size,
                                output_json_filepath=output_json_filepath,
                                output_json_filename=output_json_filename,
                                accelerator=accelerator)

     # Inference
     predictions = inferencer.inference(retriever, ice_template=template)

     if accelerator:
          if accelerator.is_main_process:
               score = AccEvaluator().score(predictions, data.references)
               with open(f"{output_json_filepath}/{args.dataset_name}_results.txt", 'a') as f:
                    f.write(f"{args.dataset_name},{args.ice_num},{args.model_name},{SBERT_CODE[args.sentence_transformers_name]},{score}\n")
               print(score)
     else:
          score = AccEvaluator().score(predictions, data.references)
          with open(f"{output_json_filepath}/{args.dataset_name}_results.txt", 'a') as f:
               f.write(f"{args.dataset_name},{args.ice_num},{args.model_name},{SBERT_CODE[args.sentence_transformers_name]},{score}\n")
          print(score)


if __name__ == '__main__':
    args = get_args()
    main(args)