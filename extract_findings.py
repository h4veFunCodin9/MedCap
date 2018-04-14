import xml.etree.ElementTree as ET
import os

def extract_id_findings(filepath):
    tree = ET.ElementTree(file=filepath)
    ids = []
    findings = None
    for elem in tree.iter('AbstractText'):
        if elem.attrib['Label'] == 'FINDINGS':
            findings = elem.text
    for elem in tree.iter('parentImage'):
        ids.append(elem.attrib['id'])
    if findings == None or ids == []:
        return None
    return [(id, findings) for id in ids]

def extract(root, dst):
    results = []
    none_cnt = 0
    for name in os.listdir(root):
        path = os.path.join(root, name)
        pairs = extract_id_findings(path)
        if pairs == None:
            none_cnt += 1
            continue
        results.extend(pairs)

    with open(os.path.join(dst, 'findings.txt'), 'w+') as f:
        for item in results:
            f.write(item[0]+"\t"+item[1]+"\n")

    print(none_cnt)


extract('../ecgen-radiology/', '../')