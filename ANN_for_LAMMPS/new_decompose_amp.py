import ast
import linecache
import os

element_list = ['Br', 'C', 'H', 'I', 'N', 'Pb']
ann_pot_name = '-'.join([temp for temp in element_list])+".ann"

fingerprint = linecache.getline('8000amp.amp', 1).strip().split('\\n')[0].split('"')[1].split('(')[1][:-1].split('=')[1]
fprange = linecache.getline('8000amp.amp', 2).strip().split('\\n')[1].strip().split('=')[1][:-1]
hidden_layer = linecache.getline('8000amp.amp', 2).strip().split('\\n')[2].strip().split('=')[1][:-1]
dict_fgp = ast.literal_eval(fingerprint)
dict_fpr = ast.literal_eval(fprange)
dict_hdl = ast.literal_eval(hidden_layer)
temp_scl = open('temp_scl.txt', 'w')
temp_wts = open('temp_wts.txt', 'w')
temp_scl.write(linecache.getline('8000amp.amp', 2).strip().split('\\n')[5].strip().split('=')[1][:-1])
temp_wts.write(linecache.getline('8000amp.amp', 2).strip().split('\\n')[7].strip().split('=')[1][:-1])
temp_scl.close()
temp_wts.close()

def split_list(input_list, length):
  return [input_list[i:i+length] for i in xrange(length,len(input_list),length)]

for i in element_list:
  output_file = open(i + "_pot", 'a')
  output_file.write("# "+i+" potential\n")
  G2_list = ['', '', '', '']
  G4_list = ['', '', '', '', '', '', '']
  for j in range(len(dict_fgp[i])):
    terms = dict_fgp[i][j]
    if terms['type']=='G2':
      G2_list.append(terms['element'])
      G2_list.append(terms['eta'])
      G2_list.append(dict_fpr[i][j][0])
      G2_list.append(dict_fpr[i][j][1])
    else:
      G4_list.append(terms['elements'][0])
      G4_list.append(terms['elements'][1])
      G4_list.append(terms['eta'])
      G4_list.append(terms['gamma'])
      G4_list.append(terms['zeta'])
      G4_list.append(dict_fpr[i][j][0])
      G4_list.append(dict_fpr[i][j][1])
  G2_list = split_list(G2_list, 4)
  G4_list = split_list(G4_list, 7)
  output_file.write('{} {}\n'.format("G2", len(G2_list)))
  for k in G2_list:
    output_file.write('{} {} {} {}\n'.format(k[0], k[1], k[2], k[3]))
  output_file.write('{} {}\n'.format("G4", len(G4_list)))
  for k in G4_list:
    output_file.write('{} {} {} {} {} {} {}\n'.format(k[0], k[1], k[2], k[3], k[4], k[5], k[6]))
  n_layer = len(dict_hdl[i])
  output_file.write('{} {}\n'.format("NLayer", n_layer))
  output_file.close()

os.system("cat wts_1 temp_wts.txt wts_2 temp_scl.txt wts_3 > new_wts.py")
os.popen("python new_wts.py")
os.system("echo '# potential for Br-C-H-I-N-Pb' > header") 
cat_command = "cat header " + ' '.join([temp+"_pot" for temp in element_list]) + " > " + ann_pot_name
os.system(cat_command)
