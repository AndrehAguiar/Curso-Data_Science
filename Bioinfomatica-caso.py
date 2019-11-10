cont_bac = {}
cont_hum = {}
comb = ['A', 'T', 'C', 'G']
saida = open("dna.html","w")

with open("data/bacteria.fasta") as file_bac:
    linhas = [k.strip() for k in file_bac]

if linhas[0].startswith('>'):
    header = linhas[0]
    seq_bac = ''.join(linhas[1:])
else:
    header = ""
    seq_bac = ''.join(linhas)
    
entrada_bac = seq_bac.replace("\n","")

for i in comb:
    for j in comb:
        cont_hum[i+j] = 0
        cont_bac[i+j] = 0

for l in range(len(entrada_bac)-1):   
    cont_bac[entrada_bac[l]+entrada_bac[l+1]] += 1

saida.write("<h2>Comparação de DNA (Bactéria / Humano)</h2><div style='color:#fff; display:inline-block'><div style='margin:0 20px 20px 0; float:left; display:grid; grid-template-columns:100px 100px 100px 100px; grid-template-rows:100px 100px 100px 100px'>")

for l in cont_bac:
  transp = cont_bac[l]/max(cont_bac.values())
  saida.write("<div style='border:1px solid #111; background-color:rgba(0, 0, 0,"+str(transp)+"')>"+l+"</div>")

saida.write("</div>")
saida.write("<div style='margin:0 20px 20px 0; float:left; display:grid; grid-template-columns:100px 100px 100px 100px; grid-template-rows:100px 100px 100px 100px'>")

# DNA Humano
with open("data/human.fasta") as file_hum:
    linhas = [k.strip() for k in file_hum]

if linhas[0].startswith('>'):
    header = linhas[0]
    seq_hum = ''.join(linhas[1:])
else:
    header = ""
    seq_hum = ''.join(linhas)
    
entrada_hum = seq_hum.replace("\n","")

for k in range(len(entrada_hum)-1):   
    cont_hum[entrada_hum[k]+entrada_hum[k+1]] += 1

for k in cont_hum:
  transp = cont_hum[k]/max(cont_hum.values())
  saida.write("<div style='border:1px solid #111; background-color:rgba(0, 0, 0,"+str(transp)+"')>"+k+"</div>")

saida.write("</div></div>")
saida.close()
