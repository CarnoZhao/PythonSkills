using DelimitedFiles, Statistics, Random
using Flux
using CUDAdrv, CuArrays
using PyPlot

function load_data(train_ratio, use_gpu = true)
    data = readdlm("/home/tongxueqing/zhaox/codes/PythonSkills/LianJia/contents.csv", '\t',  Float64, header = true)[1]
    data = data'
    data[9, :] = log.(data[9, :])
    data = (data .- mean(data, dims = 2)) ./ std(data, dims = 2)
    newOrder = shuffle(1:size(data)[2])
    data = data[:, newOrder]
    train_size = floor(Int, train_ratio * size(data)[2])
    X, Y = data[1:size(data)[1] - 1, 1:train_size], data[size(data)[1], 1:train_size]
    tX, tY = data[1:size(data)[1] - 1, train_size + 1:size(data)[2]], data[size(data)[1], train_size + 1:size(data)[2]]
    if use_gpu
        X |> gpu, reshape(Y, 1, :) |> gpu, tX |> gpu, reshape(tY, 1, :) |> gpu
    else
        X, reshape(Y, 1, :), tX, reshape(tY, 1, :)
    end
end

function predict(X)
    Yhat = net(X)
    [yhat.data for yhat in Yhat]
end

function model(numIters; learnRate = 0.0001, beta1 = 0.9, beta2 = 0.999)
    X, Y, tX, tY = load_data(0.9)
    net = Chain(
        Dense(17, 50, relu, initW = Flux.glorot_normal),
        Dense(50, 30, relu, initW = Flux.glorot_normal),
        Dense(30, 10, relu, initW = Flux.glorot_normal),
        Dense(10, 5, relu, initW = Flux.glorot_normal),
        Dense(5, 1, initW = Flux.glorot_normal)
    ) |> gpu
    loss(x, y) = ((net(x) - y) * (net(x) - y)' / size(y)[2])[1, 1]
    parameters = params(net)
    data = Iterators.repeated((X, Y), numIters)
    callback = () -> @show(loss(X, Y))
    optimizer = AdaMax(learnRate, (beta1, beta2))
    Flux.train!(loss, parameters, data, optimizer, cb = Flux.throttle(callback, 5))
    println("Loss in test: $(loss(tX, tY))")
    net
end

net = model(3000);

X, Y, tX, tY = load_data(0.9, false)
PyPlot.scatter(1:size(tY)[2], predict(tX) - tY)
PyPlot.savefig("/home/tongxueqing/zhaox/codes/PythonSkills/LianJia/estimate.png")