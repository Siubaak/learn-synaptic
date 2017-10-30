const { Layer, Network, Trainer } = require('synaptic')

const ip = new Layer(2)
const h1 = new Layer(3)
const op = new Layer(1)

ip.project(h1)
h1.project(op)

const net = new Network({
  input: ip,
  hidden: [h1],
  output: op
})

const trainingSet = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
]

const trainer = new Trainer(net)
trainer.train(trainingSet, {
	rate: 0.3,
  error: 0.00001,
  iterations: 10000,
  shuffle: true,
  log: 1000,
	cost: Trainer.cost.MSE
})

console.log('[0, 0]: ' + net.activate([0, 0]))
console.log('[0, 1]: ' + net.activate([0, 1]))
console.log('[1, 0]: ' + net.activate([1, 0]))
console.log('[1, 1]: ' + net.activate([1, 1]))