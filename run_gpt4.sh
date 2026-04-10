#Fallacy=['Hasty Generalization','Equivocation','Slippery Slope','Appeal to Authority','False Dilemma','Ad Hominem','Straw Man','Red Herring','False Causality','Circular Reasoning']
for fallacy in 'Hasty Generalization'
    do
        python eval_COT.py --model_name "gpt-4" --dataset_name "NQ1" --begin 0 --Fallacy "${fallacy}" 
    done

    