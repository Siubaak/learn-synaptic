const { Layer, Network, Trainer, Neuron } = require('synaptic')

const ip = new Layer(1)
const h1 = new Layer(16)
const op = new Layer(1)

ip.set({
  squash: Neuron.squash.RELU
})
h1.set({
  squash: Neuron.squash.RELU
})
op.set({
  squash: Neuron.squash.IDENTITY
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
  const x = Math.random() * 2 - 1
  trainingSet.push({
    input: [x],
    output: [x*x]
  })
}

const trainer = new Trainer(net)
trainer.train(trainingSet, {
	rate: 0.02,
  error: 0.00001,
  iterations: 20000,
  shuffle: true,
  log: 4000,
	cost: Trainer.cost.MSE
})

console.log('0.33*0.33 = ' + net.activate([0.33]) + ' should be ~ ' + (0.33 * 0.33).toFixed(3))
console.log('0.66*0.66 = ' + net.activate([0.66]) + ' should be ~ ' + (0.66 * 0.66).toFixed(3))