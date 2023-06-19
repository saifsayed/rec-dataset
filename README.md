# A New Dataset and Approach for Timestamp Supervised Action Segmentation Using Human Object Interaction

Here is the code for our CVPR 2023 paper : A New Dataset and Approach for Timestamp Supervised Action Segmentation Using Human Object Interaction
 * [Project Page](https://saifsayed.github.io/Rec.github.io/)
 * [Paper](https://openaccess.thecvf.com/content/CVPR2023W/LSHVU/papers/Sayed_A_New_Dataset_and_Approach_for_Timestamp_Supervised_Action_Segmentation_CVPRW_2023_paper.pdf)
 * [Video](https://drive.google.com/file/d/1k10xURqylCnFMexyg-GDV9OcXwIx7soz/view?usp=sharing)

 Abstract: This paper focuses on leveraging Human Object Interaction (HOI) information to improve temporal action segmentation under timestamp supervision, where only one frame is annotated for each action segment. This information is obtained from an off-the-shelf pre-trained HOI detector, that requires no additional HOI-related annotations in our experimental datasets. Our approach generates pseudo labels by expanding the annotated timestamps into intervals and allows the system to exploit the spatio-temporal continuity of human interaction with an object to segment the video. We also propose the (3+1)Real-time Cooking (ReC)1 dataset as a realistic collection of videos from 30 participants cooking 15 breakfast items. Our dataset has three main properties: 1) to our knowledge, the first to offer syn- chronized third and first person videos, 2) it incorporates diverse actions and tasks, and 3) it consists of high resolution frames to detect fine-grained information. In our experi- ments we benchmark state-of-the-art segmentation methods under different levels of supervision on our dataset. We also quantitatively show the advantages of using HOI information, as our framework improves its baseline segmentation method on several challenging datasets with varying viewpoints, providing improvements of up to 10.9% and 5.3% in F1 score and frame-wise accuracy respectively.


Main Software Requirements
	Python 3.6
	numpy and others
	PyTorch 1.1

********************************************************************************************************************
## Training on GTEA
This section illustrates the instructions to reproduce the action segmentation using Timestamp supervision results on the GTEA dataset using I3D features.

The code will create the new pseudo ground truths using HOI and timestamps, then train all the splits and eventually produce a consolidated result for all test splits.

Make sure you have the following files and folders accordingly

	| rec-dataset
		| models
		| results
		| runs
		| data
			| gtea
				| features
				| groundTruth
				| transcripts
				| bbox_obj_interact
				| splits
				| mapping.txt
				| gtea_annotation_all.npy
		| batch_gen.py
		| consolidate_accuracies.py
		| create_hoi_gts_gtea.py
		| eval.py
		| get_accuracies.py
		| main.py
		| model.py
		| Readme.md
		| run_exps.sh

### Data Preparation
* Download the pre-computed I3D features from the third party link used in [1]:  [Link](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8)
* Extract the content and copy the contents of "ms_tcn_data/gtea" folder to "code/data/gtea" directory.
### Execution
Go to code directory and type the following command line in terminal

	chmod u+x run_exps.sh 
	./run_exps.sh


# (3+1) Real-time Cooking (ReC) Dataset
This section provides the information for the new dataset we recorded called (3+1) Real-time Cooking (ReC) Dataset

ReC Data Download: [Link](https://drive.google.com/drive/folders/1usqia6A0cg415oCMDtKu0Z00KGP1_KXu?usp=share_link)

After downloading the folder arrangments for the dataset are as follows. More information of the individual folders can be found below.

	| Videos
	| splits_c123
	| splits_c4
	| groundTruth
	| features.tar.gz
	| bbox_obj_interact_all
	| mapping



### Videos
* The videos compressed in 3 parts for 30 subjects. Once extracted, Each video is named P<Subject_idx>_<Dish_Name>_C<1/2/3/4>. 
* Each Subject name starts with an identifier P<Subject_number>
* For example the "Videos" folder arrangement once extracted for a single subject P20 and dishname: PeanutButterSandwich will be as follows:

		| Videos	
			| P01
			.
			.
			| P20
				| PeanutButterSandwich
					| P20_PeanutButterSandwich_C1.mp4
					| P20_PeanutButterSandwich_C2.mp4
					| P20_PeanutButterSandwich_C3.mp4
					| P20_PeanutButterSandwich_C4.mp4
					| Readme.txt
				| OrangeJuice
				.
				.
			| P30

* The folder for a specific dishname (For eg. Tea) will have 4 videos (3 for Thirdperson view and 1 for Ego-centric). C4 stands for Ego-centric while C1/C2/C3 will stand for Third-person views.
* It will also have an optional Readme.txt which will illustrate which camera index are in sync (< 1sec) if all 4 views aren't in sync.

### Split Files
* splits_c123: Filename splits for all third-person view (C1/C2/C3 reserved for third-person view)
* splits_c4: Filename splits for ego-centric view. (C4 reserved for ego-centric view)


### HOI Detections
bbox_obj_interact_all.tar.gz: Output of HOI detector

### Other Folders
* groundTruth: Frame Level Ground truth for all videos
* features.tar.gz: I3D features for all videos
* mapping: action to label number mapping


### Dataset Information
* We provide [Sync_Table](https://docs.google.com/spreadsheets/d/1yDYp3S9cGpLVMxBLcSLJ47njzwWARjAhjJ6Mc1cU32I/edit?usp=sharing) which illustrats the camera indices which are in sync if for a specific dish any of the 4 cameras are out of sync (>1sec). Each individual sheet is a Subject.
* We provide [DataRecord](https://docs.google.com/spreadsheets/d/1FCFD7YW_sfelpWcuyrOHrkVoSNFWAIxTthw4aCnwGtE/edit?usp=sharing) which gives more information of the individual videos. It provides  the following information for every subject
	* Video Name: Name of the video
	* Resolution: Recorded video resolution
	* FPS: Recorded frame rate
	* Kitchen ID: For a kitchen, there can be different camera configuration. We capture this information using a unique identifier of the kitchen in the form of (KitchenID).(Camera_configuration_id) 
	* Frame Loss? (Y/N): Yes means there is frame loss
	* Approximate frame loss time if available: Approximate frame loss duration.

Please create an issue for any concerns.


[1] Y. Abu Farha and J. Gall.
MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

[2] A. Richard, H. Kuehne, A. Iqbal, J. Gall:
NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning
in IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2018


## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{sayed2023new,
        title={A New Dataset and Approach for Timestamp Supervised Action Segmentation Using Human Object Interaction},
        author={Sayed, Saif and Ghoddoosian, Reza and Trivedi, Bhaskar and Athitsos, Vassilis},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={3132--3141},
        year={2023}
      }
```