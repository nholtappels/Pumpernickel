import os

def slice_csv(csv_name, slice_size):
    size_of_slice = slice_size  # number of tweets in one slice
    infile_name = csv_name
    outfile_name_raw = csv_name.replace('.csv', '')

    infile = open(infile_name)
    slice_names = []

    i = 0
    h = 1
    filenumber = 0
    outfile = None
    print 'Start slicing'
    for line in infile:
        if h:
            h = 0
        else:
            if i % size_of_slice == 0:
                outfile_name = outfile_name_raw + '-' + str(filenumber) + '.csv'
                slice_names.append(outfile_name)
                print 'write file %s' % outfile_name
                outfile = open(outfile_name, 'w')
                filenumber += 1
            outfile.write(line)
            i += 1
    print str(filenumber) + ' slices created'

    return slice_names

def merge_csvs(csv_names):
    print 'Merging csv files'
    outfile_name = csv_names[0].replace('-0', '')
    outfile = open(outfile_name, 'w')
    for csv_name in csv_names:
        infile = open(csv_name)
        for line in infile:
            outfile.write(line)
        infile.close()
        os.remove(csv_name)
    print outfile_name + "created"
