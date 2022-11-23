chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/19PatchSize 

python3 train.py --patchSize 19 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 19 --device cuda  --softLabels   >> experiments/19PatchSize/PolyUexp49.log; wait;
python3 train.py --patchSize 19 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 19 --device cuda  --softLabels   >> experiments/19PatchSize/PolyUexp50.log; wait;
python3 train.py --patchSize 19 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 19 --device cuda  --softLabels   >> experiments/19PatchSize/PolyUexp51.log; wait;
python3 train.py --patchSize 19 --poreRadius 6 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp52.log; wait;

python3 train.py --patchSize 19 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --maxPooling True --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp53.log; wait;
python3 train.py --patchSize 19 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --maxPooling True --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp54.log; wait;
python3 train.py --patchSize 19 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --maxPooling True --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp55.log; wait;
python3 train.py --patchSize 19 --poreRadius 6 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --maxPooling True --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp56.log; wait;

python3 train.py --patchSize 19 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp57.log; wait;
python3 train.py --patchSize 19 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp58.log; wait;
python3 train.py --patchSize 19 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp59.log; wait;
python3 train.py --patchSize 19 --poreRadius 6 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp60.log; wait;

python3 train.py --patchSize 19 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --maxPooling True --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp61.log; wait;
python3 train.py --patchSize 19 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --maxPooling True --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp62.log; wait;
python3 train.py --patchSize 19 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --maxPooling True --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp63.log; wait;
python3 train.py --patchSize 19 --poreRadius 6 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --maxPooling True --experimentPath experiments/19PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 19 --device cuda --softLabels   >> experiments/19PatchSize/PolyUexp64.log; wait;