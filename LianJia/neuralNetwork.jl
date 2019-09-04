using DelimitedFiles, Statistics, Random
using Flux
using CUDAdrv, CuArrays, CUDAnative
using Plots

function load_data(train_ratio, use_gpu = true)
    data = readdlm("/home/tongxueqing/zhaox/codes/PythonSkills/LianJia/contents.csv", '\t',  Float64, header = true)[1]
    data = permutedims(data, (2, 1))
    data = data[:, data[size(data)[1], :] .< 10000]
    data = data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 18], :]
    data[9, :] = log.(data[9, :]) # use log of time span
    # data = (data .- mean(data, dims = 2)) ./ std(data, dims = 2)
    μ = mean(data, dims = 2)
    σ = std(data, dims = 2)
    newOrder = shuffle(1:size(data)[2])
    data = data[:, newOrder]
    train_size = floor(Int, train_ratio * size(data)[2])
    X, Y = data[1:size(data)[1] - 1, 1:train_size], data[size(data)[1], 1:train_size]
    tX, tY = data[1:size(data)[1] - 1, train_size + 1:size(data)[2]], data[size(data)[1], train_size + 1:size(data)[2]]
    X = (X .- μ[1:length(μ) - 1]) ./ σ[1:length(σ) - 1]
    tX = (tX .- μ[1:length(μ) - 1]) ./ σ[1:length(σ) - 1]
    Y = Y / 1000
    tY = tY / 1000
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
    push!(trloss, mean(Float64.((net(X) - Y).data)))
    push!(tsloss, mean(Float64.((net(tX) - tY).data)))
    # println("Train loss: $(round(Float64(loss(X, Y).data), digits = 4))")
    # println("Test loss: $(round(Float64(loss(tX, tY).data), digits = 4))")
    return
end

function predict_plot(tX, tY, net)
    order = sortperm(tY[:])
    typlot = tY[order]
    tYhat = predict(net, tX[:, order])[:]
    Plots.scatter(1:length(typlot), typlot, markersize = 2)
    scatter!(1:length(tYhat), tYhat, markersize = 2, markercolor = :red)
end

function model(numIters; learnRate = 5e-3, beta1 = 0.9, beta2 = 0.999)
    trloss = []
    tsloss = []
    X, Y, tX, tY = load_data(0.9)
    net = Chain(
        Dense(11, 20, tanh, initW = Flux.glorot_normal),
        Dense(20, 10, tanh, initW = Flux.glorot_normal),
        Dense(10, 5, tanh, initW = Flux.glorot_normal),
        Dense(5, 1, initW = Flux.glorot_normal)
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

net, trloss, tsloss = model(6000);
Plots.plot([x for x in trloss][200:length(trloss)], label = "Train");
plot!([x for x in tsloss][200:length(tsloss)], label = "Test")

X, Y, tX, tY = load_data(0.9);
predict_plot(X, Y, net)
predict_plot(tX, tY, net)

#   l1  l2  l3  lr      it      func    apx     loss
#   20  10  5   5e-3    16000   relu            0.368
#   20  10  5   5e-3    8000    tanh            0.304
#   20  10  5   5e-3    4000    tanh            0.271
#   20  10  5   5e-3    4000    tanh    clean   0.249
#   20  10  5   5e-3    6000    tanh    clean   0.238