
def prepare_one_dataset(dataset_name):
  dataset_name = dataset_name.lower()

  if dataset_name == 'modelnet10':
      pre_processed_path = 'ModelNet10/processed/'
      processed_path = 'ModelNet10_walks/3DFutureModels/'
      if not os.path.isdir(p_out):
          os.makedirs(p_out)


if __name__ == '__main__':
  utils.config_gpu(False)
  np.random.seed(1)

  if len(sys.argv) != 2:
    print('Use: python dataset_prepare.py <dataset name>')
    print('For example: python dataset_prepare.py modelnet40_normal_resampled')
    print('Another example: python dataset_prepare.py all')
  else:
    dataset_name = sys.argv[1]
    prepare_one_dataset(dataset_name)
