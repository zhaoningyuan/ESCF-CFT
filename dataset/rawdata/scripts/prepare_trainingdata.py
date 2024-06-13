import os
import json

prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request..."

rawTrainingDataFolder = "../train"
processedTrainingDataFolder = "../../train"
def typeCheck(inputs, targets):
    if type(inputs) == list and len(inputs) == 1:
        inputs = inputs[0]
    if type(targets) == list and len(targets) == 1:
        targets = targets[0]
    return inputs, targets

if __name__ == "__main__":
    absoluteRawTrainingDataFolder = os.path.abspath(rawTrainingDataFolder)
    absoluteProcessedTrainingDataFolder = os.path.abspath(processedTrainingDataFolder)
    for taskName in os.listdir(absoluteRawTrainingDataFolder):
        taskPath = os.path.join(absoluteRawTrainingDataFolder, taskName)
        datasetName = os.listdir(taskPath)[0]
        print(f"Task: {taskName}")
        print(f"Dataset: {datasetName}")
        datasetPath = os.path.join(taskPath, datasetName)
        output_dir = os.path.join(absoluteProcessedTrainingDataFolder, taskName)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # For emdg and eSNLI
        if taskName == "emdg" or taskName == "exp":
            files = os.listdir(datasetPath)
            for file in files:
                if "train" in file:
                    trainFile = file
                if "test" in file:
                    testFile = file

            with open(os.path.join(datasetPath, trainFile)) as f:
                trainData = json.load(f)
            content = []
            for inputs, targets in zip(trainData["src"], trainData["tgt"]):
                inputs, targets = typeCheck(inputs, targets)
                inputs = prefix + inputs
                content.append({"prompt": inputs, "response": targets})
            with open(os.path.join(absoluteProcessedTrainingDataFolder, taskName, "train.jsonl"), "w") as f:
                for line in content:
                    f.write(json.dumps(line) + "\n")

            with open(os.path.join(datasetPath, testFile)) as f:
                testData = json.load(f)
            content = []
            for inputs, targets in zip(testData["src"], testData["tgt"]):
                inputs, targets = typeCheck(inputs, targets)
                inputs = prefix + inputs
                content.append({"prompt": inputs, "response": targets})
            with open(os.path.join(absoluteProcessedTrainingDataFolder, taskName, "test.jsonl"), "w") as f:
                for line in content:
                    f.write(json.dumps(line) + "\n")
            
        # For  eli5 and wiki_auto
        elif taskName == "inqqg" or taskName == "simp":
            files = os.listdir(datasetPath)
            trainFiles = []
            for file in files:
                if "train" in file:
                    trainFiles.append(file)
                if "test" in file:
                    testFile = file
            content = []
            for trainFile in trainFiles:
                with open(os.path.join(datasetPath, trainFile)) as f:
                    trainData = json.load(f)
                for inputs, targets in zip(trainData["src"], trainData["tgt"]):
                    inputs, targets = typeCheck(inputs, targets)
                    inputs = prefix + inputs
                    content.append({"prompt": inputs, "response": targets})
            with open(os.path.join(absoluteProcessedTrainingDataFolder, taskName, "train.jsonl"), "w") as f:
                for line in content:
                    f.write(json.dumps(line) + "\n")

            with open(os.path.join(datasetPath, testFile)) as f:
                testData = json.load(f)
            content = []
            for inputs, targets in zip(testData["src"], testData["tgt"]):
                inputs, targets = typeCheck(inputs, targets)
                inputs = prefix + inputs
                content.append({"prompt": inputs, "response": targets})
            with open(os.path.join(absoluteProcessedTrainingDataFolder, taskName, "test.jsonl"), "w") as f:
                for line in content:
                    f.write(json.dumps(line) + "\n")
            
        # For gigaword
        elif taskName == "hgen":
            files = os.listdir(datasetPath)
            trainFiles = []
            testFiles = []
            for file in files:
                if file.startswith("constrain") and file.endswith("train.json"):
                    trainFiles.append(file)
                if file.startswith("constrain") and file.endswith("title.test.json"):
                    testFiles.append(file)
            content = []
            for trainFile in trainFiles:
                with open(os.path.join(datasetPath, trainFile)) as f:
                    trainData = json.load(f)
                for inputs, targets in zip(trainData["src"], trainData["tgt"]):
                    inputs, targets = typeCheck(inputs, targets)
                    inputs = prefix + inputs
                    content.append({"prompt": inputs, "response": targets})
            with open(os.path.join(absoluteProcessedTrainingDataFolder, taskName, "train.jsonl"), "w") as f:
                for line in content:
                    f.write(json.dumps(line) + "\n")
            
            content = []
            for testFile in testFiles:
                with open(os.path.join(datasetPath, testFile)) as f:
                    testData = json.load(f)
                for inputs, targets in zip(testData["src"], testData["tgt"]):
                    inputs, targets = typeCheck(inputs, targets)
                    inputs = prefix + inputs
                    content.append({"prompt": inputs, "response": targets})
            with open(os.path.join(absoluteProcessedTrainingDataFolder, taskName, "test.jsonl"), "w") as f:
                for line in content:
                    f.write(json.dumps(line) + "\n")
        else:
            print(f"Task {taskName} not recognized. Skipping...")
            
    

