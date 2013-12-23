
size_of_slice = 5000   #number of tweets in one slice
infile_name = '../data/test.csv'
outfile_name_raw = '../data/test'

infile = open(infile_name)

i = 0
filenumber = 0
outfile = None
for line in infile:
    if i % size_of_slice == 0:
        outfile_name = outfile_name_raw + str(filenumber) + '.csv'
        print 'write file %s' % outfile_name
        outfile = open(outfile_name, 'w')
        filenumber += 1
    outfile.write(line)
    i += 1


