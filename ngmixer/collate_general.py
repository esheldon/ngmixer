from .collate import *

class ConcatGeneral(Concat):
    """
    This class is a general version of Concat
    It replaces the I/O in Concat with chunks based on a file with the structure
    
    #cstart cend cfile
    #comments
    0 59 /path/to/chunk/file/chunk.fits
    ....
    
    You specify this file and the output options with keywords
    chunk_file : the path to the chunk file above
    out_dir : output dir
    out_file : output file name base
        This name will be formatted like <self.run>_self.out_file-blind.fits
        if the code is to blind or <self.run>_<self.out_file>.fits 
        otherwise.
    
    """
    def __init__(self, *args, **kwargs):
        #process the args for this class
        self.chunk_file = kwargs.pop('chunk_file', None)
        assert self.chunk_file is not None,"chunk_file must be given!"
        self.out_dir = kwargs.pop('out_dir', None)
        assert self.out_dir is not None,"out_dir must be given!"
        self.out_file = kwargs.pop('out_file', None)
        assert self.out_file is not None,"out_file must be given!"        

        #init the parent
        super(ConcatGeneral, self).__init__(*args, **kwargs)

    def set_chunks(self):
        """
        set the chunks in which the meds file was processed
        """
        self.chunk_list = []
        self.split_files = []
        for line in open(self.chunk_file,'r'):
            if line[0] != '#':
                line = line.strip()
                line = line.split()
                self.chunk_list.append([int(line[0]),int(line[1])])
                self.split_files.append(line[2])

    def make_collated_dir(self):
        """
        set collated file output dir
        """
        files.try_makedir(self.out_dir)

    def set_collated_file(self):
        """
        set the output file and the temporary directory
        """
        if self.blind:
            extra='-blind'
        else:
            extra=''
            
        self.collated_file = os.path.join(self.out_dir, "%s_%s%s.fits" % (self.run,self.out_file,extra))
        self.tmpdir=files.get_temp_dir()

    def read_chunk(self, split):
        """
        read data and epoch data from a given split
        """
        ind = 0
        for ind,tsplit in enumerate(self.chunk_list):
            if set(tsplit) == set(split):
                break
        chunk_file=self.split_files[ind]

        print(chunk_file)
        data, epoch_data, meta=self.read_data(chunk_file, split)
        return data, epoch_data, meta

    def set_files(self, run, root_dir):
        pass
