

#Generate pseudo ground truths using timestamps and HOI
python create_hoi_gts_gtea.py


#Train and evaluate
for split_id in 1 2 3 4
do
  python main.py --split $split_id --action train --dataset 'gtea' --jump 75 --threshold 10
  python main.py --split $split_id --action predict --dataset 'gtea' --jump 75 --threshold 10
  python get_accuracies.py --split $split_id --action predict --dataset 'gtea' --jump 75 --threshold 10
done

#Consolidate Results
python consolidate_accuracies.py