x0 = [[0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 2, 0],
      [0, 1, 0, 2, 0, 1, 0],
      [0, 1, 0, 2, 2, 0, 0],
      [0, 2, 0, 0, 2, 0, 0],
      [0, 2, 1, 2, 2, 0, 0],
      [0, 0, 0, 0, 0, 0, 0]]
x1 = [[0, 0, 0, 0, 0, 0, 0],
      [0, 2, 1, 2, 1, 1, 0],
      [0, 2, 1, 2, 0, 1, 0],
      [0, 0, 2, 1, 0, 1, 0],
      [0, 1, 2, 2, 2, 2, 0],
      [0, 0, 1, 2, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0]]
x2 = [[0, 0, 0, 0, 0, 0, 0],
      [0, 2, 1, 1, 2, 0, 0],
      [0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 1, 0, 2, 1, 0, 0],
      [0, 2, 2, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0]]
f0_w0 = [[-1, 0, 1], [0, 0, 1], [1, -1, 1]]
f0_w1 = [[-1, 0, 1], [1, -1, 1], [0, 1, 0]]
f0_w2 = [[-1, 1, 1], [1, 1, 0], [0, -1, 0]]
f1_w0 = [[0, 1, -1], [0, -1, 0], [0, -1, 1]]
f1_w1 = [[-1, 0, 0], [1, -1, 0], [1, -1, 0]]
f1_w2 = [[-1, 1, -1], [0, -1, -1], [1, 0, 0]]
r0 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
r1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
b0 = 1
b1 = 0

inputs = [x0, x1, x2]
filters0 = [f0_w0, f0_w1, f0_w2]
filters1 = [f1_w0, f1_w1, f1_w2]


def One(input, filter, s, e):
	result = 0
	for row in range(len(filter)):
		for col in range(len(filter[0])):
			result += input[s + row][e + col] * filter[row][col]
	return result


def padding(inputs, output, filters, b, pad):
	for r in range(len(output)):
		for c in range(len(output[0])):
			add = 0
			for i in range(len(inputs)):
				add += One(inputs[i], filters[i], r * pad, c * pad)
			add += b
			output[r][c] = add
	return output


print(padding(inputs, r0, filters0, b0, 2))
print(padding(inputs, r1, filters1, b1, 2))
