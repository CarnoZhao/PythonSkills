using DelimitedFiles, Statistics, Random
using Flux
using CUDAdrv, CuArrays, CUDAnative
using Plots

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

function predict(net, X)
    Yhat = net(X)
    [yhat.data for yhat in Yhat] |> gpu
end

function callback_func(trloss, tsloss, loss, X, Y, tX, tY)
    push!(trloss, Float64(loss(X, Y).data))
    push!(tsloss, Float64(loss(tX, tY).data))
    # println("Train loss: $(round(Float64(loss(X, Y).data), digits = 4))")
    # println("Test loss: $(round(Float64(loss(tX, tY).data), digits = 4))")
    return
end

function model(numIters; learnRate = 0.0001, beta1 = 0.9, beta2 = 0.999)
    trloss = []
    tsloss = []
    X, Y, tX, tY = load_data(0.9)
    net = Chain(
        Dense(17, 30, relu, initW = Flux.glorot_normal),
        Dense(30, 10, relu, initW = Flux.glorot_normal),
        Dense(10, 1, tanh, initW = Flux.glorot_normal)
    ) |> gpu
    parameters = params(net)
    loss(x, y) = ((net(x) - y) * (net(x) - y)' / size(y)[2])[1, 1]
    data = Iterators.repeated((X, Y), numIters)
    callback = () -> callback_func(trloss, tsloss, loss, X, Y, tX, tY)
    optimizer = AdaMax(learnRate, (beta1, beta2))
    Flux.train!(loss, parameters, data, optimizer, cb = Flux.throttle(callback, 0.1))
    println("Loss in test: $(Float64(loss(tX, tY).data))")
    net, trloss, tsloss
end

net, trloss, tsloss = model(4000);
Plots.plot([x for x in trloss], label = "Train");
plot!([x for x in tsloss], label = "Test")