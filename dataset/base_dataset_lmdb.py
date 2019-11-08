import os
import os.path as ops
import lmdb

class BaseDatsetLmdb(object):

    def __init__(self):
        self.lmdb_path = None
        self.datset_list = None
        self.image_data_db_names = None
        self.annot_path = None
        self.pase = 'train'
    def get_annot(self, data_list):
        raise NotImplementedError
    
    def get_images(self, data_list):
        raise NotImplementedError
    
    def _process_dir(self):
        raise NotImplementedError
    def GetLmdbPath(self):
        return self.lmdb_path
    def generate_lmdb(self):
        self.lmdb_path = ops.join(self.lmdb_path, self.pase+'_lmdb')
        print(' Now generated lmdb: lmdb data to {}'.format(self.lmdb_path))
        self._write_lmdb()

    def _write_lmdb(self):
        if ops.exists(self.lmdb_path):
            print("LMDB existes :{}".format(self.lmdb_path))
            return
        self._process_dir()
        assert isinstance(self.datset_list, list)
        assert isinstance(self.datset_list[0], list)
        size_db = 0
        for i in self.datset_list:
            size_db += len(i)
        max_map_size = int(size_db * 1224*1224*3)

        num_dbs = len(self.datset_list[0]) +1

        evn = lmdb.open(self.lmdb_path,max_dbs= num_dbs, map_size=max_map_size)
        dbs = {}
        for name in self.image_data_db_names:
            dbs[name] = evn.open_db(name.encode())
        dbs_annot = evn.open_db('annot'.encode())
        generated_index = 0
        for data_list in self.datset_list:
            image_paths = self.get_images(data_list)
            paths = {}
            for name in self.image_data_db_names:
                if not os.path.isfile(image_paths[name]):
                    print('skip {}'.format(image_paths[name]))
                    paths.clear()
                    break
                paths[name] = image_paths[name]
            for name in self.image_data_db_names:
                path = paths[name]
                with open(path, 'rb') as imgf:
                    imgb = imgf.read()
                with evn.begin(write=True) as txn:
                    txn.put(str(generated_index).encode(), imgb, db=dbs[name])
            annot = self.get_annot(data_list)
            if annot is not None:
                with evn.begin(write=True) as txn:
                    txn.put(str(generated_index).encode, annot.encode(), db=dbs_annot)
            print('process image {}'.format(generated_index))
            generated_index += 1
