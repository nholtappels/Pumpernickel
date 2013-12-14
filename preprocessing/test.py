'''
Created on 14.12.2013

@author: Nick
'''

delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
print delchars
