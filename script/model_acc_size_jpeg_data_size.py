with open('../resource/analysis/arg_vs_jpeg_file_size.tsv', 'r') as fp:
	arg_dict = dict()
	for line in fp:
		elements = line.strip().split('\t')
		img_size, crop_pct, interpolaton = elements[0].split(' ')[1], elements[1].split(' ')[1], elements[2].split(' ')[1]
		crop_pct = '{:.3f}'.format(float(crop_pct))
		key = f'{img_size}-{crop_pct}-{interpolaton}'
		arg_dict[key] = elements[-1].split(' ')[0]

line_list = list()
with open('../resource/analysis/results-imagenet.csv', 'r') as fp:
	next(fp)
	for line in fp:
		elements = line.strip().split(',')
		model_name, top1_acc, param_count, img_size = elements[0], elements[1], elements[5], elements[6]
		if len(elements) == 0:
			continue
		arg_str = f'{img_size}-{elements[-2]}-{elements[-1]}'
		jpeg_file_size_kb = arg_dict[arg_str]
		line_list.append(f'{model_name}\t{top1_acc}\t{param_count}\t{jpeg_file_size_kb}\t{img_size}')
		
with open('../resource/analysis/offload_cost_vs_model_acc_size.tsv', 'w') as fp:
	fp.write('model name\ttop1 acc\t#params [M]\tjpeg file size [KB]\tinput size\n')
	for line in line_list:
		fp.write('{}\n'.format(line))

