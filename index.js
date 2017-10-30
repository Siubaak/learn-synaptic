const { Layer, Network, Trainer, Neuron } = require('synaptic')

const ip = new Layer(1)
const h1 = new Layer(10)
const op = new Layer(1)

ip.set({
  squash: Neuron.squash.RELU
})
h1.set({
  squash: Neuron.squash.RELU
})
op.set({
  squash: Neuron.squash.RELU
})

ip.project(h1)
h1.project(op)

const net = new Network({
  input: ip,
  hidden: [h1],
  output: op
})

const trainingSet = []
for (let i = 0; i < 100; i ++) {
  const x = Math.random()
  trainingSet.push({
    input: [x],
    output: [x*x]
  })
}

const trainer = new Trainer(net)
trainer.train(trainingSet, {
	rate: 0.02,
  error: 0.00008,
  iterations: 100000,
  shuffle: true,
  log: 5000,
	cost: Trainer.cost.MSE
})
console.log(net.activate([0.33])[0])
console.log(net.activate([0.67])[0])

