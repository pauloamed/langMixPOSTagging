from posembd.base import get_batches

def loadDatasets(datasets, datasetsDir):
    # Loading datasets from datasets folder
    datasetsPreparer = DatasetsPreparer(datasetsDir)
    datasets = datasetsPreparer.prepare(datasets)

    # Retrieving tags and char dicts
    tagsFromDatasets = [(dataset.name, dataset.id2tag) for dataset in datasets]
    char2id, id2char = datasetsPreparer.getDicts()

    return datasets, tagsFromDatasets, id2char, char2id


def prepLoop(itr, device, model):
    # Getting vars
    inputs, targets, datasetName = itr

    # Setting the input and the target (seding to GPU if needed)
    inputs = [[word.to(device) for word in sample] for sample in inputs]
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

    # Feeding the model
    output = model(inputs)

def trainLog(currentLR, name2TrainLoss, name2ValLoss, datasets, duration):
    # Verbose
    outStr = "\n======================================================================================="
    totalTrainLoss = sum([name2TrainLoss[d.name] for d in datasets if d.useTrain])
    totalValLoss = sum([name2ValLoss[d.name] for d in datasets if d.useVal])
    outStr += ("Epoch: {} \t Learning Rate: {:.3f}\tTotal Training Loss: {:.6f} \tTotal Validation Loss: {:.6f} \t Duration: {:.3f}\n".format(
        epoch, currentLR, totalTrainLoss, totalValLoss, duration))

    for d in datasets:
        if d.useTrain and d.useVal:
            outStr += ('>> Dataset {}:\tTraining Loss: {:.6f}\tValidation Loss:{:.6f}\n'.format(d.name, name2TrainLoss[d.name], name2ValLoss[d.name]))
        elif d.useTrain and not d.useVal:
            outStr += ('>> Dataset {}:\tTraining Loss: {:.6f}\n'.format(d.name, name2TrainLoss[d.name]))
        elif not d.useTrain and d.useVal:
            outStr +=('>> Dataset {}:\tValidation Loss: {:.6f}\n'.format(d.name, name2ValLoss[d.name]))

    outStr += ("----------------------------------------------------------------------------------------\n")

    # Saving the best model
    outStr += ('Comparing loss on {} dataset(s)\n'.format([d.name for d in datasets if d.useVal]))

    return outStr


def train(device, model, modelPath, datasets, parameters, minValLoss=np.inf):
    batchSize = parameters['batchSize']
    epochs = parameters['epochs']
    gradClipping = parameters['gradClipping']

    # optimizer and loss function
    optimizer = torch.optim.Adadelta(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    name2dataset = {d.name: d for d in datasets}

    for epoch in range(epochs):
        start = time.time()

        name2TrainLoss = {d.name : 0.0 for d in datasets}
        name2ValLoss = {d.name : 0.0 for d in datasets}

        model.train()
        for itr in get_batches(datasets, "train", batchSize, "visconde"):
            inputs, output, targets, datasetName = prepLoop(itr, device, model)

            # Reseting the gradients
            optimizer.zero_grad()

            # Calculating the loss and the gradients
            loss = criterion(output[datasetName].view(batchSize * output["length"], -1),
                             targets.view(batchSize * output["length"]))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradClipping)

            # Adjusting the weights
            optimizer.step()

            # Updating the train loss
            name2TrainLoss[datasetName] += loss.item() * batchSize


        model.eval()
        for itr in get_batches(datasets, "val"):
            inputs, outputs , targets, datasetName = prepLoop(itr, device, model)

            # Calculating the loss and the gradients
            loss = criterion(output[datasetName].view(output["length"], -1),
                             targets.view(output["length"]))

            # Updating the loss accu
            name2ValLoss[datasetName] += loss.item()

        duration = time.time() - start
        outStr = trainLog(optimizer.param_groups[0]['lr'], name2TrainLoss, name2ValLoss, datasets, duration)

        if totalValLoss <= minValLoss:
            torch.save(model.state_dict(), modelPath)
            outStr += ('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(
                                                                                minValLoss,
                                                                                totalValLoss))
            minValLoss = totalValLoss
        outStr += "=======================================================================================\n"

        send_output(outStr, 0)

    return model, minValLoss


def accuracy(device, model, datasets):
    name2dataset = {d.name: d for d in datasets}

    name2correctPreds = {d.name : 0 for d in datasets}
    name2totalPreds = {d.name : 0 for d in datasets}

    model.eval()
    for itr in get_batches(datasets, "test"):
        inputs, output, targets, datasetName = prepLoop(itr, device, model)

        # convert output probabilities to predicted class
        _, pred = torch.max(output[datasetName], 2)

        # Formatando vetor
        pred = pred.view(1, -1)

        # calculate test accuracy for each object class
        for i in range(output["length"]):
            if i >= len(targets[0]):
                break
            if targets.data[0][i].item() <= 1:
                continue

            label, predicted = targets.data[0][i], pred.data[0][i]
            name2correctPreds[datasetName] += 1 if label == predicted else 0
            name2totalPreds[datasetName] += 1

    correctPredsSum = np.sum([name2correctPreds[d.name] for d in datasets])
    totalPredsSum = np.sum([name2totalPreds[d.name] for d in datasets])
    acc = 100. * soma_correct / soma_total
    outStr = '\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (acc, correctPredsSum, totalPredsSum)

    send_output(outStr, 0)

    for d in datasets:
        datasetAccuracy = 100. * name2correctPreds[d.name] / name2totalPreds[d.name]
        outStr = '\nTest Accuracy (on {} Dataset): {:.2f}% ({}/{})'.format(d.name, datasetAccuracy,
                                                                        name2correctPreds[d.name], name2totalPreds[d.name])
        send_output(outStr, 0)

def tagged_samples(device, model, datasets, id2char):
    name2dataset = {d.name:d for d in datasets if d.useVal == True}
    name2taggedSamples = {d.name:([],[]) for d in datasets if d.useVal == True}


    model.eval()
    for itr in get_batches(datasets, "val"):
        inputs, output, targets, datasetName = prepLoop(itr, device, model)

        # convert output probabilities to predicted class
        _, pred = torch.max(output[datasetName], 2)

        # Formatando vetor
        pred = pred.view(1, -1)

        # lists storing the current sample
        writtenInputs, goldTags, predTags = [], [], []

        # boolean storing if the sentence had one wrong labeled word
        mistagged = False

        for i in range(output["length"]):
            # Esse loop inteiro assume que BATCH_SIZE = 1

            if i >= len(targets[0]):
                break
            if targets.data[0][i].item() <= 1:
                continue

            label, predicted = targets.data[0][i], pred.data[0][i]

            if label != predicted:
                mistagged = True

            writtenInputs.append("".join([id2char[charid] for charid in inputs[0][i]]))
            goldTags.append(name2dataset[datasetName].id2tag[targets.data[0][i].item()])
            predTags.append(name2dataset[datasetName].id2tag[pred.data[0][i].item()])

        if mistagged:
            name2taggedSamples[datasetName][1].append((writtenInputs, goldTags, predTags))
        else:
            name2taggedSamples[datasetName][0].append((writtenInputs, goldTags, predTags))


    for datasetName, (correctSamples, mistaggedSamples) in name2taggedSamples.items():

        file = open("tagged_samples_{}".format(datasetName),"w", encoding="utf-8")
        file.write("\n\n================================================================\n")
        file.write("====================  MISTAGGED SAMPLES  =======================\n")
        file.write("================================================================\n\n")

        for sampleInput, sampleGoldTags, samplePredTags in mistaggedSamples:
            file.write("\n\n\n")
            file.write("{}\n".format(" ".join(sampleInput)))
            file.write("(token, gold_label, pred_label)\n")
            for i in range(len(sampleInput)):
                if sampleGoldTags[i] != samplePredTags[i]:
                    file.write(">>>>> ")
                file.write("(\'{}\', {}, {})\n".format(sampleInput[i],
                                                   sampleGoldTags[i],
                                                   samplePredTags[i]))



        file.write("\n\n================================================================\n")
        file.write("======================  CORRECT SAMPLES  =======================\n")
        file.write("================================================================\n\n")

        for sampleInput, sampleGoldTags, samplePredTags in correctSamples:
            file.write("\n\n\n")
            file.write("{}\n".format(" ".join(sampleInput)))
            file.write("(token, gold_label, pred_label)\n")
            for i in range(len(sampleInput)):
                file.write("(\'{}\', {}, {})\n".format(sampleInput[i],
                                                   sampleGoldTags[i],
                                                   samplePredTags[i]))

        file.close()
