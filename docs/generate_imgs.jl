using EvalMetrics
using Plots; pyplot()
using Random, MLBase; Random.seed!(42);
scores = sort(rand(10000));
targets = scores .>= 0.99;
targets[MLBase.sample(findall(0.98 .<= scores .< 0.99), 30; replace = false)] .= true;
targets[MLBase.sample(findall(0.99 .<= scores .< 0.995), 30; replace = false)] .= false;

savefig(prplot(targets, scores), "pr1.png")
savefig(prplot([targets, targets], [scores, scores .+ rand(10000) ./ 5]), "pr2.png")
savefig(rocplot(targets, scores), "roc1.png")
savefig(rocplot([targets, targets], [scores, scores .+ rand(10000) ./ 5]), "roc2.png")

savefig(prplot(targets, scores; xguide="RECALL", fill=:green, grid=false, xlims=(0.8, 1.0)), "pr3.png")
savefig(rocplot(targets, scores, title="Title", label="experiment", xscale=:log10), "roc3.png")
savefig(rocplot([targets, targets], [scores, scores .+ rand(10000) ./ 5], label=["a" "b";]), "roc4.png")

prplot(targets, scores; npoints=Inf, label="Original") 
prplot!(targets, scores; npoints=10, label="Sampled (10 points)") 
prplot!(targets, scores; npoints=100, label="Sampled (100 points)") 
prplot!(targets, scores; npoints=1000, label="Sampled (1000 points)") 
prplot!(targets, scores; npoints=5000, label="Sampled (5000 points)") 
savefig("pr4.png")

savefig(rocplot(targets, scores; aucshow=false, label="a", diagonal=true), "roc5.png")
