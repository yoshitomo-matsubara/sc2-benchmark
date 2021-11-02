with open('results-imagenet.csv', 'r') as fp:
	arg_str_set = set()
	next(fp)
	for line in fp:
		elements = line.strip().split(',')
		if len(elements) == 0:
			continue
		arg_str = f'--img_size {elements[-3]} --crop_pct {elements[-2]} --interpolation {elements[-1]}'
		arg_str_set.add(arg_str)
		

for arg_str in list(arg_str_set):
	print(arg_str)

