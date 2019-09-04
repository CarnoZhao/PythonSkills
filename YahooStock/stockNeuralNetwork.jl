using DelimitedFiles, Statistics, Random
using Flux
using CUDAdrv, CuArrays, CUDAnative
using Plots

function load_data(preSpan = 5, postSpan = 5, trainRatio = 0.9, useGPU = true)
    data = readdlm("/data/tongxueqing/zhaox/stockCompiled/total.csv", ',',  header = true)[1]
    data = data[:, [!("" in data[:, col]) for col in 1:size(data)[2]]]
    dates = data[:, 1]
    data = permutedims(Float64.(data[:, 2:size(data)[2]]), (2, 1))
    before = data[:, preSpan:size(data)[2] - postSpan]
    after = data[:, preSpan + postSpan:size(data)[2]]
    totalY = after .> before
    rowMu = mean(data, dims = 2)
    rowStd = std(data, dims = 2)
    data = (data .- rowMu) ./ rowStd
    totalX = zeros(size(data)[1], preSpan, 1, size(data)[2] - preSpan - postSpan + 1)
    for col in 1:size(totalX)[4]
        totalX[:, :, :, col] = data[:, col:col + preSpan - 1]
    end
    newOrder = shuffle(1:size(totalX)[4])
    totalX = totalX[:, :, :, newOrder]
    totalY = totalY[:, newOrder]
    dataSize = length(newOrder)
    trainSize = floor(Int, dataSize * trainRatio)
    X = totalX[:, :, :, 1:trainSize]
    Y = totalY[:, 1:trainSize]
    tX = totalX[:, :, :, trainSize + 1:dataSize]
    tY = totalY[:, trainSize + 1:dataSize]
    if useGPU
        X = X |> gpu
        Y = Y |> gpu
        tX = tX |> gpu
        tY = tY |> gpu
    end
    X, Y, tX, tY
end

function callback_func(x, y, tx, ty, loss, trloss, tsloss)
    push!(trloss, Float64(loss(x, y).data))
    push!(tsloss, Float64(loss(tx, ty).data))
end

function model(numIter, learnRate = 0.0003, beta1 = 0.9, beta2 = 0.999; preSpan = 5, postSpan = 5, trainRatio = 0.9, useGPU = true)
    trloss = []
    tsloss = []
    X, Y, tX, tY = load_data(preSpan, postSpan, trainRatio, useGPU)
    # convnet = Chain(
    #     Conv((1, 3), 1 => 4, pad = (0, 0), relu),
    #     MaxPool((1, 2)),
    #     Conv((1, 3), 4 => 8, pad = (0, 0), relu),
    #     MaxPool((1, 2)),
    #     x -> reshape(x, :, size(x)[4]),
    #     Dense(2944, 500, relu, initW = Flux.glorot_normal),
    #     Dense(500, 368, sigmoid, initW = Flux.glorot_normal)
    # ) |> gpu
    convnet = Chain(
        x -> reshape(x, :, size(x)[4]),
        Dense(1840, 800, tanh, initW = Flux.glorot_normal),
        #Dense(900, 500, tanh, initW = Flux.glorot_normal),
        Dense(800, 368, sigmoid, initW = Flux.glorot_normal)
    ) |> gpu
    loss(x, y) = -mean(sum(y .* log.(convnet(x)) + (1 .- y) .* log.(1 .- convnet(x)), dims = 1))
    accuracy(x, y) = mean((convnet(x) .> 0.5) .== y)
    parameters = params(convnet)
    data = Iterators.repeated((X, Y), numIter)
    callback = () -> callback_func(X, Y, tX, tY, loss, trloss, tsloss)
    optimizer = Flux.AdaMax(learnRate, (beta1, beta2))
    Flux.train!(loss, parameters, data, optimizer, cb = Flux.throttle(callback, 0.1))
    println("Accuracy in training: $(round(accuracy(X, Y) * 100, digits = 4))%")
    println("Accuracy in test: $(round(accuracy(tX, tY) * 100, digits = 4))%")
    convnet, trloss, tsloss
end

convnet, trloss, tsloss = model(4000, preSpan = 5, postSpan = 5);
plot(1:length(trloss), [x for x in trloss], label = "Train");
plot!(1:length(tsloss), [x for x in tsloss], label = "Test");
savefig("/home/tongxueqing/zhaox/codes/PythonSkills/YahooStock/out.png");

# 1840 -> 500 -> 368 for 4000, lr = 1e-3, 83.4955% 75.8982%
# 1840 -> 700 -> 400 -> 368 for 4000, 81.7041% 76.4296%
# 1840 -> 700 -> 400 -> 368 for 6000, 84.2778% 76.8048%
# 1840 -> 700 -> 400 -> 368 for 6000, lr = 3e-3, 85.733% 77.2954%
# 1840 -> 700 -> 368 for 4000, lr = 3e-3, 87.2758% 79.3526%
