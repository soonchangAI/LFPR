import torch
import argparse
import numpy as np 
import os 

def calc_full_params_tap(num_layers):

    full_parameters = 0
    for i in range(num_layers):
        num_heads = 12
        num_neurons = 3072
        att_params = (num_heads * 64) *768 *3 +  (num_heads * 64) *3 + 768 * (num_heads * 64) + 768 + 768 * 2
        ffn_params = (num_neurons * 768) + num_neurons + 768 * num_neurons  + 768 + 768 * 2
        full_parameters += (att_params + ffn_params)
    return full_parameters

    
"""
Adapted from A Fast Post-Training Pruning Framework for Transformers
"""
def prune_constrained_sum_of_importance(head_importance,ffn_importance,constraint,num_layers):

    num_attention_heads = 12
    intermediate_size = 3072
    hidden_size = 768
    attention_head_size = 64
    full_params = calc_full_params_tap(num_layers)
     

    max_params = constraint * full_params
    sorted_head_importance, sorted_head_indicies = head_importance.view(-1).sort(descending=True) 
    sorted_ffn_importance, sorted_ffn_indicies = ffn_importance.view(-1).sort(descending=True)
    cfg = []
    importances =  []
    max_importance = 0
    for num_heads in range(1, num_layers * num_attention_heads + 1): 

        head_params = (num_heads * 64) *768 *3 +  (num_heads * 64) *3 + 768 * (num_heads * 64) + 768 + 768 * 2
        ffns_params = max_params-head_params  
        num_ffns = int(ffns_params / 1537)
        num_ffns = max(num_ffns, 0)


        total_importance = sorted_head_importance[:num_heads].sum() + sorted_ffn_importance[:num_ffns].sum()

        cfg.append([num_heads,num_ffns])
        importances.append(total_importance)
        if total_importance > max_importance:
            max_importance = total_importance
            head_indicies = sorted_head_indicies[:num_heads] 
            ffn_indicies = sorted_ffn_indicies[:num_ffns] 
    head_mask = torch.zeros(num_layers * num_attention_heads).cuda()
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_layers, num_attention_heads)

    ffn_mask = torch.zeros(num_layers * intermediate_size).cuda()
    ffn_mask[ffn_indicies] = 1.0
    ffn_mask = ffn_mask.view(num_layers, intermediate_size)

    cfg_head = []
    for i in range(num_layers):
        cfg_head.append(int(head_mask[i,:].sum()))

    cfg_ffn = []
    for i in range(num_layers):
        cfg_ffn.append(int(ffn_mask[i,:].sum()))

    return head_mask,ffn_mask,cfg_head,cfg_ffn


def prune_threshold(head_importance,ffn_importance,percentile,num_layers):
    print(f"Percentile : {percentile}")
    
    head_importance_vec = head_importance.view(-1)
    ffn_importance_vec = ffn_importance.view(-1)
    all_scores = torch.cat([head_importance_vec, ffn_importance_vec])
    thre = np.percentile(all_scores, percentile)
    head_mask = head_importance.gt(thre).float().cuda()
    ffn_mask = ffn_importance.gt(thre).float().cuda()


    cfg_head = []
    for i in range(num_layers):
        cfg_head.append(int(head_mask[i,:].sum()))

    cfg_ffn = []
    for i in range(num_layers):
        cfg_ffn.append(int(ffn_mask[i,:].sum()))

    return head_mask,ffn_mask,cfg_head,cfg_ffn

    
def create_pruned_model(head_mask, ffn_mask, model, num_layers):

    # indices of head 
    head_idxs = []
    for i in range(num_layers):
        head_idxs.append(torch.nonzero(head_mask[i]).squeeze(-1).tolist())
    # indices of filters
    ffn_idxs = []
    for i in range(num_layers):
        ffn_idxs.append(torch.nonzero(ffn_mask[i]).squeeze(-1).tolist())
    # indices of head dimensions
    head_indices = []
    for i in range(num_layers):
        idxs = head_idxs[i]
        indices = []
        for j in idxs:
            indices += np.arange(j*64, j*64 + 64).tolist()
        head_indices.append(indices)

    i = 0
    pruned_model_dict = {}

    for k,v in model.items():
        if i == num_layers:
            i = 0
        if 'mmt.encoder.layer.' + str(i) +'.attention.self.query.weight' in k:
            idx = head_indices[i]
            pruned_model_dict[k] = v[idx].clone() 
        elif 'mmt.encoder.layer.' + str(i) +'.attention.self.query.bias' in k:
            idx = head_indices[i]
            pruned_model_dict[k] = v[idx].clone() 
        elif 'mmt.encoder.layer.' + str(i) +'.attention.self.key.weight' in k:
            idx = head_indices[i]
            pruned_model_dict[k] = v[idx].clone() 
        elif 'mmt.encoder.layer.' + str(i) +'.attention.self.key.bias' in k:
            idx = head_indices[i]
            pruned_model_dict[k] = v[idx].clone() 
        elif 'mmt.encoder.layer.' + str(i) +'.attention.self.value.weight' in k:
            idx = head_indices[i]
            pruned_model_dict[k] = v[idx].clone() 
        elif 'mmt.encoder.layer.' + str(i) +'.attention.self.value.bias' in k:
            idx = head_indices[i]
            pruned_model_dict[k] = v[idx].clone() 
        elif 'mmt.encoder.layer.'+str(i)+'.attention.output.dense.weight' in k:
            idx =  head_indices[i]
            pruned_model_dict[k] = v[:,idx].clone()   
            # add head mask
            mask_k = 'mmt.encoder.layer.'+str(i)+'.attention.self.head_mask.mask'
            mask_val = torch.ones(1,len(head_idxs[i]),1,1).cuda()
            pruned_model_dict[mask_k] = mask_val   

        elif 'mmt.encoder.layer.'+str(i)+'.intermediate.dense.weight' in k:
            idx = ffn_idxs[i]
            pruned_model_dict[k] = v[idx].clone() 
            # add ffn mask
            mask_k = 'mmt.encoder.layer.'+str(i)+'.intermediate.ffn_mask.mask'
            mask_val = torch.ones(len(idx)).cuda()
            pruned_model_dict[mask_k] = mask_val   

        elif 'mmt.encoder.layer.'+str(i)+'.intermediate.dense.bias' in k:
            idx = ffn_idxs[i]
            pruned_model_dict[k] = v[idx].clone()  
        elif 'mmt.encoder.layer.' + str(i) +'.output.dense.weight' in k:
            idx = ffn_idxs[i]
            pruned_model_dict[k] = v[:,idx].clone() 
            i = i + 1

        else:
            pruned_model_dict[k] = v.clone()
    
    for k,v in pruned_model_dict.items():
        print(k,v.shape)
    return pruned_model_dict

def main():
    parser = argparse.ArgumentParser(description="LFPR pruning")
    
    parser.add_argument("--prune_constraint", type=float,
                        help="For LFPR, the percentage of remaining parameters (0.0 to 1.0)")

    parser.add_argument("--output_dir", type=str,
                        help="Directory to save the pruned model")
    parser.add_argument("--model",type=str, help="Directory of the original model")

    parser.add_argument("--score_dir",type=str,help='Directory that saves the LFPR scores')

    parser.add_argument('--num_layers',type=int)

    parser.add_argument('--heuristic',type=str)

    parser.add_argument('--percentile',type=float, help="For LFRP(thre), the percentile value for pruning by threshold (0-100)")
    
    args = parser.parse_args()
    constraint = args.prune_constraint
    percentile = args.percentile
    
    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)
    score_dir = args.score_dir
    num_layers = args.num_layers

    model = torch.load(args.model)['model']

    # Head and FFN grads
    head_grads = torch.tensor(np.load(score_dir + "/head_grads.npy"))
    ffn_grads = torch.tensor(np.load(score_dir + "/ffn_grads.npy"))

    head_importance = head_grads.pow(2).sum(dim=0)
    ffn_importance = ffn_grads.pow(2).sum(dim=0)
    if args.heuristic == "sum":
        head_mask, ffn_mask, cfg_head, cfg_ffn = prune_constrained_sum_of_importance(head_importance,ffn_importance,constraint,num_layers)
    elif args.heuristic == "thre":
        head_mask,ffn_mask,cfg_head,cfg_ffn = prune_threshold(head_importance,ffn_importance,percentile,num_layers)
    pruned_dict = create_pruned_model(head_mask, ffn_mask, model, num_layers)
    pruned_ckpt = {'model':pruned_dict}
    torch.save(pruned_ckpt , output_dir + "/pruned_model.ckpt")
    print(output_dir + "/pruned_model.ckpt")

    
    # Write configuration to cfg.txt
    with open(output_dir + "/cfg.txt","w") as f:
        f.write("[" +",".join([str(x) for x in cfg_head]) + "]")
        f.write("\n")
        f.write("[" +",".join([str(x) for x in cfg_ffn]) + "]")
        f.write("\n")
if __name__ == "__main__":
    main()