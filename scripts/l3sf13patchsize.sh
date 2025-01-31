chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/13PatchSize 

 #python3 train.py --patchSize 13 --poreRadius 3 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 13 --device cuda  --softLabels >> experiments/13PatchSize/L3SFexp1.log; wait;
 #python3 train.py --patchSize 13 --poreRadius 4 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 13 --device cuda  --softLabels >> experiments/13PatchSize/L3SFexp2.log; wait;
 #python3 train.py --patchSize 13 --poreRadius 5 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 13 --device cuda  --softLabels >> experiments/13PatchSize/L3SFexp3.log; wait;
 #python3 train.py --patchSize 13 --poreRadius 6 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 13 --device cuda --softLabels >> experiments/13PatchSize/L3SFexp4.log; wait;

 #python3 train.py --patchSize 13 --poreRadius 3 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 13 --device cuda --softLabels >> experiments/13PatchSize/L3SFexp5.log; wait;
 #python3 train.py --patchSize 13 --poreRadius 4 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 13 --device cuda --softLabels >> experiments/13PatchSize/L3SFexp6.log; wait;
 python3 train.py --patchSize 13 --poreRadius 5 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 13 --device cuda --softLabels >> experiments/13PatchSize/L3SFexp7.log; wait;
 python3 train.py --patchSize 13 --poreRadius 6 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 13 --device cuda --softLabels >> experiments/13PatchSize/L3SFexp8.log; wait;

python3 train.py --patchSize 13 --poreRadius 3 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 13 --device cuda --softLabels     >> experiments/13PatchSize/L3SFexp9.log; wait;
python3 train.py --patchSize 13 --poreRadius 4 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 13 --device cuda --softLabels   >> experiments/13PatchSize/L3SFexp10.log; wait;
python3 train.py --patchSize 13 --poreRadius 5 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 13 --device cuda --softLabels   >> experiments/13PatchSize/L3SFexp11.log; wait;
python3 train.py --patchSize 13 --poreRadius 6 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 13 --device cuda --softLabels   >> experiments/13PatchSize/L3SFexp12.log; wait;

python3 train.py --patchSize 13 --poreRadius 3 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 13 --device cuda --softLabels   >> experiments/13PatchSize/L3SFexp13.log; wait;
python3 train.py --patchSize 13 --poreRadius 4 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 13 --device cuda --softLabels   >> experiments/13PatchSize/L3SFexp14.log; wait;
python3 train.py --patchSize 13 --poreRadius 5 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 13 --device cuda --softLabels   >> experiments/13PatchSize/L3SFexp15.log; wait;
python3 train.py --patchSize 13 --poreRadius 6 --groundTruth  dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/13PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 13 --device cuda --softLabels   >> experiments/13PatchSize/L3SFexp16.log; wait;
