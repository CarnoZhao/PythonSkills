using DelimitedFiles, Statistics, Random
using Flux
using CUDAdrv, CuArrays, CUDAnative

function load_data(preSpan = 10, trainRatio = 0.9, useGPU = true)
    data = readdlm("/data/tongxueqing/zhaox/stockCompiled/total.csv", ',',  header = true)[1]
    data = data[:, [!("" in data[:, col]) for col in 1:size(data)[2]]]
    dates = data[:, 1]
    data = permutedims(Float64.(data[:, 2:size(data)[2]]), (2, 1))
    before = data[:, preSpan:size(data)[2] - 1]
    after = data[:, preSpan + 1:size(data)[2]]
    totalY = after .> before
    rowMu = mean(data, dims = 2)
    rowStd = std(data, dims = 2)
    data = (data .- rowMu) ./ rowStd
    totalX = zeros(size(data)[1], preSpan, 1, size(data)[2] - preSpan)
    for col in 1:size(totalX)[4]
        totalX[:, :, 1, col] = data[:, col:col + preSpan - 1]
    end
    newOrder = shuffle(1:size(totalX)[4])
    totalX = totalX[:, :, 1, newOrder]
    totalY = totalY[:, newOrder]
    dataSize = length(newOrder)
    trainSize = floor(Int, dataSize * trainRatio)
    X = totalX[:, :, 1, 1:trainSize]
    Y = totalY[:, :, 1:trainSize]
    tX = totalX[:, :, 1, trainSize + 1:dataSize]
    tY = totalY[:, :, trainSize + 1:dataSize]
    if useGPU
        X = X |> gpu
        Y = Y |> gpu
        tX = tX |> gpu
        tY = tY |> gpu
    end
    X, Y, tX, tY
end

function model(numIter, learnRate = 0.0001, beta1 = 0.9, beta2 = 0.999; preSpan = 10, trainRatio = 0.9, useGPU = true)
    X, Y, tX, tY = load_data(preSpan, trainRatio, useGPU)
    convnet = Chain(
        Conv((size(X)[1], 3), 1 => 4, pad = (0, 0), relu),
        MaxPool((1, 2)),
        Conv((size(X)[1], 3), 4 => 8, pad = (0, 0), relu),
        MaxPool((1, 2))
        x -> reshape(x, :, size(x)[4]),
        Dense(2944, 500, relu, initW = Flux.glorot_normal),
        Dense(500, 368, sigmoid, initW = Flux.glorot_normal)
    )
    loss(x, y) = mean(sum(y .* log.(convnet(x)), dims = 1))
    accuracy(x, y) = mean((convnet(x) .> 0.5) .== y)
    parameters = params(convnet)
    data = Iterators.repeated((X, Y), numIter)
    callback = () -> @show(loss(X, Y))
    optimizer = Flux.AdaMax(learnRate, (beta1, beta2))
    Flux.train!(loss, parameters, data, optimizer, cb = Flux.throttle(callback, 1))
    println("Accuracy in training: $(round(accuracy(X, Y) * 100, digits = 4))%")
    println("Accuracy in test: $(round(accuracy(tX, tY) * 100, digits = 4))%")
    convnet
end

convnet = model(100);
