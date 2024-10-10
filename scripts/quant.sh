
Model_Path=""
Saving_Path=""
Precision_Path=""


##### fake quantization to test the performance of MC-MoE #####
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path}


##### real quantization and model pack for compact storage #####
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}