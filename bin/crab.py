import math
import matplotlib.pyplot as plt
import numpy as np

E=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1, 2,3,4,5,6,7,8,9,10]

q=[1.05,1.36,2.22,2.64,3.25,3.81,4.01,4.34,4.54,5.00, 5.41,5.58,5.60,5.61,5.62,5.60,5.40,5.2,5]

d=[0.33224598722791143, 0.3005661524582392, 0.26915889987022384, 0.24099130744514555, 0.24889246844256874, 0.24858157453160296, 0.23107783294736645, 0.24040003463768553, 0.24705188551071325, 0.242636921507013,      0.25477685929173155, 0.26998411423082547, 0.29734225574076845, 0.3083781841285265, 0.31071358442217095, 0.33804888822233636, 0.3451913522647237, 0.35778800367852215, 0.37401256137562766]

eff_g=[0.6357786070207584, 0.8549184654276746, 0.9052752230985455, 0.9211866946359005, 0.9233488300610904, 0.9271826439456498, 0.9300939530648565, 0.9295134978018385, 0.9315373861790154, 0.932354297470824,      0.9339902156687108, 0.9358134369794558, 0.6734228362877998, 0.7282862139317983, 0.7634006773327106, 0.7861915367483296, 0.8127598566308244, 0.8362411998775635, 0.8537775534241966]
eff_p=[0.10969759483930121, 0.3114567088315139, 0.49784873591010914, 0.6342970187068337, 0.7231312819457308, 0.7886795029359552, 0.8304922951043229, 0.8607409925220938, 0.8804868926592091, 0.888998369122044,      0.9225452676072429, 0.9315435214930397, 0.43466404060298464, 0.4868631661136053, 0.533378966017072, 0.5746119990734306, 0.6119727739343896, 0.6414272181875592, 0.6668778710597181]

DetectG=[]
DetectP=[]
# cs=[0.9999469,0.9999761,0.9999846,0.9999877,0.9999901,0.9999906,0.9999913,0.9999930,0.9999938,0.9999942]
# lg=[-1,-0.6990,-0.5229,-0.3979,-0.301,-0.2218,-0.1549,-0.0969,-0.04576,0]
for i,eng in enumerate(E):
    DetectG.append(2.83/1.62*1e-11*math.pow(eng,-1.62)*eff_g[i]*800*800*10000*0.68*1.152e6)
    DetectP.append(0.898*1e-5*math.pow(10,-(math.log10(eng)+1)/0.67)*eff_p[i]*800*800*10000*1.152e6*2*math.pi*(1-math.cos(d[i]*math.pi/180)))

# Q=[1.6, 3.108888542538125, 3.719652475210671, 3.9799780858579865, 4.215272387789213, 4.290366313937477, 4.355447716599306, 4.440554166234005, 4.480604260179746, 4.540679401098356]
Sensitivity=[]
for i,eng in enumerate(E):
    Sigma=DetectG[i]*q[i]/math.sqrt(DetectP[i])
    Sensitivity.append(2.83/1.62*1e-11*math.pow(eng,-1.62)*(5/Sigma)*eng)

print(Sensitivity)



plt.figure(figsize=(8,6))

ARGO_YBJ_x=[3.76304635626435e-001,4.82451029424275e-001,5.46273422968356e-001,6.34100759520183e-001,7.36048572597984e-001,8.33418831552485e-001,9.67412090477020e-001,1.18016422652229e+000,1.43970456362210e+000,1.80051060394572e+000,2.36646845348959e+000,2.88689942940949e+000,3.43535184009192e+000,4.19085040088231e+000,4.98702708156546e+000,6.08376824757209e+000,7.99609044493859e+000,1.05095164381310e+001,1.31433186136399e+001,1.77093017855467e+001]
ARGO_YBJ_y=[1.93069772888325e-011,1.66256724792430e-011,1.51991108295293e-011,1.43167405882297e-011,1.34855955305256e-011,1.27027018260320e-011,1.16127496875312e-011,1.09385823063689e-011,1.03035530854230e-011,9.41945930179987e-012,8.87262135382639e-012,8.35752957426481e-012,8.61122496314302e-012,9.14195251260080e-012,9.14195251260080e-012,9.14195251260080e-012,9.14195251260080e-012,9.14195251260080e-012,9.14195251260080e-012,9.41945930179987e-012]
plt.plot(ARGO_YBJ_x,ARGO_YBJ_y,label="ARGO_YBJ",linestyle="--")

Fermi_4y_x=[1.07762808537615e-003,1.21924724906278e-003,1.41405843059167e-003,1.63999651970872e-003,2.04802853784438e-003,2.31717528027978e-003,2.62169260843944e-003,3.19494007601977e-003,3.89353124637800e-003,5.10989995386488e-003,7.40443427988027e-003,9.96284979732399e-003,1.27617568147439e-002,1.44435485439943e-002,1.80517112461307e-002,2.31267996831346e-002,2.75121742788548e-002,3.27291170382762e-002,3.89353124637800e-002,4.63258416745566e-002,5.51281288288090e-002,6.24335182129399e-002,7.42963950759495e-002,8.62511684099974e-002,1.10625208128180e-001,1.56734572714317e-001]
Fermi_4y_y=[1.07815136869232e-011,1.01516339087623e-011,9.55855309468529e-012,9.00012136815300e-012,7.97922633632089e-012,7.51306235781444e-012,7.07413270575738e-012,6.66084629080916e-012,6.27170497857321e-012,5.73020203610120e-012,5.23545279739997e-012,4.92958611278051e-012,4.92958611278051e-012,4.92958611278051e-012,5.08021804691302e-012,5.23545279739997e-012,5.56029763284582e-012,5.90529815896437e-012,6.27170497857321e-012,6.86437984043976e-012,7.74263682681127e-012,8.73326162382844e-012,9.85063103129499e-012,1.11109612759058e-011,1.41360084917001e-011,1.96841944728661e-011]
plt.plot(Fermi_4y_x,Fermi_4y_y,label="Fermi_4y",linestyle="--")

CTA_x=[1.95835712267272e-002,2.57519909355888e-002,3.93182875570577e-002,6.63165037146649e-002,1.93413367515095e-001,5.36697694554048e-001,1.16108727353838e+000,4.03093138316657e+000,8.09297330967955e+000,1.26678438877645e+001,2.41982402231629e+001,5.10632094842449e+001,9.05225452733553e+001]
CTA_y=[1.87381742286039e-012,1.16127496875312e-012,6.38549847031294e-013,2.45250775662011e-013,7.19685673001152e-014,3.51119173421513e-014,2.31012970008316e-014,1.71303498970736e-014,2.11190960161090e-014,2.76412985387058e-014,5.33669923120631e-014,1.12706263472941e-013,2.31012970008316e-013]
plt.plot(CTA_x,CTA_y,label="CTA",linewidth=2,linestyle="-.")

HESS_x=[1.11853260612001e-001,1.50792026718210e-001,2.13663200621497e-001,3.10378687213172e-001,5.23502691556830e-001,8.82969995549409e-001,1.25111519095362e+000,1.77275471298539e+000,2.38989256623105e+000,4.79823021356309e+000,7.14585260722924e+000,9.16562663564108e+000]
HESS_y=[1.43167405882297e-012,7.41531953658575e-013,3.95733300700463e-013,2.04969061070664e-013,1.30882962592820e-013,1.19652582873207e-013,1.19652582873207e-013,1.34855955305256e-013,1.61358633681077e-013,2.84803586843580e-013,4.20123159961896e-013,5.33669923120631e-013]
plt.plot(HESS_x,HESS_y,label="H.E.S.S.",linestyle="--")

MAGIC_x=[8.50608522508284e-002,9.87630730247652e-002,1.17562894508885e-001,1.39941313509368e-001,1.75082703173572e-001,2.60745136283188e-001,4.73887960971765e-001,9.28041863717907e-001,1.31497930778035e+000,1.86324631193156e+000,2.21792035125063e+000,2.64010756548457e+000,3.14265927241190e+000,3.74087307335290e+000,4.34348011739246e+000,5.85554829897900e+000,6.97016795732887e+000,8.29695852083491e+000]
MAGIC_y=[1.27027018260320e-012,9.14195251260080e-013,6.57933224657568e-013,4.73505115577500e-013,3.61777504265841e-013,2.45250775662011e-013,1.71303498970736e-013,1.38949549437314e-013,1.38949549437314e-013,1.56604845283283e-013,1.56604845283283e-013,1.51991108295293e-013,1.51991108295293e-013,1.66256724792430e-013,1.76503469536369e-013,2.04969061070664e-013,2.31012970008316e-013,2.52695438627470e-013]
plt.plot(MAGIC_x,MAGIC_y,label="MAGIC",linestyle="--")

LHAASO_x=[3.18201851781573e-001,4.73887960971765e-001,8.19433120276665e-001,1.41693964377027e+000,2.99003089435879e+000,8.72048301491229e+000,1.29871397400366e+001,1.79495702056665e+001,4.98077928707042e+001,7.99286913971357e+001,1.25111519095362e+002,2.11020342856860e+002,4.68026312950116e+002,6.63165037146648e+002]
LHAASO_y=[2.52695438627470e-012,1.09385823063689e-012,3.95733300700463e-013,1.47513296661055e-013,5.83759237848860e-014,6.38549847031294e-014,7.64041384905857e-014,5.33669923120631e-014,1.12706263472941e-014,7.41531953658575e-015,8.11130830789687e-015,1.34855955305256e-014,3.02356619120548e-014,4.32876128108306e-014]
plt.plot(LHAASO_x,LHAASO_y,label="LHAASO",linestyle="--")

HAWC_x=[6.18538741636995e-001,8.12965175621495e-001,1.47592656017206e+000,2.61376288673909e+000,5.11249732199352e+000,9.51518564994833e+000,1.16077541549539e+001,1.48820195425943e+001,2.21474505666158e+001,3.82590174905877e+001,6.44692351687135e+001,1.11368556326865e+002]
HAWC_y=[3.20991480968312e-012,2.17601726918124e-012,1.09385823063689e-012,6.98483005847109e-013,4.32876128108306e-013,2.68269579527973e-013,2.45250775662011e-013,2.68269579527973e-013,3.72759372031494e-013,6.01479429628180e-013,9.14195251260080e-013,1.56604845283283e-012]
plt.plot(HAWC_x,HAWC_y,label="HAWC",linestyle="--")

Milagro_x=[3.18857856533517e+000,3.52178297717650e+000,3.88980703601588e+000,4.19085040088231e+000,4.62879157571242e+000,4.98702708156546e+000,5.50816820808969e+000,6.39374624036382e+000,7.79985143933658e+000,1.02515932606259e+001,1.25061116665206e+001,1.56402829015481e+001,2.05565221382512e+001,2.50772884565825e+001,3.29598792151018e+001,4.33202193991172e+001,5.69371445975384e+001,6.77540486759327e+001,7.86472340033008e+001]
Milagro_y=[1.16127496875312e-011,1.00000000000000e-011,8.87262135382639e-012,8.11130830789687e-012,7.41531953658576e-012,6.77904990692279e-012,6.19737523296391e-012,5.33669923120631e-012,4.59555176405500e-012,3.84074597781544e-012,3.40774847773881e-012,3.02356619120548e-012,2.60366086634239e-012,2.31012970008316e-012,2.24207094478062e-012,2.11190960161090e-012,2.17601726918124e-012,2.38025439990192e-012,2.76412985387057e-012]
plt.plot(Milagro_x,Milagro_y,label="Milagro",linestyle="--")

HADAR_x=[0.031623,0.038019,0.045709,0.054954,0.066069,0.079433,0.095499,0.114815,0.138038,0.165959,0.199526,0.239883,0.288403,0.346737,0.416869,0.501187,0.602560,0.724436,0.870964,1.047129,1.258925,1.513561,1.819701,2.187762,2.630268,3.162278,3.801894,4.570882,5.495409,6.606934,7.943282,9.549926]
HADAR_y=[2.022430e-12,1.705962e-12,1.415172e-12,1.158983e-12,9.394207e-13,7.741860e-13,6.630477e-13,5.733126e-13,4.955704e-13,4.255727e-13,3.668589e-13,3.138382e-13,2.701578e-13,2.333092e-13,2.019521e-13,1.761944e-13,1.528534e-13,1.356678e-13,1.238197e-13,1.182810e-13,1.138116e-13,1.081196e-13,1.055414e-13,1.010960e-13,9.978595e-14,9.783736e-14,9.830343e-14,1.021770e-13,1.036378e-13,1.142477e-13,1.277295e-13,1.972777e-13]
HADAR_y_new=[]
for i in HADAR_y:
    HADAR_y_new.append(i+1e-13)
plt.plot(HADAR_x,HADAR_y_new,label="HADAR(Traditional)",linewidth=2,color="orange")

Deep_Learning_x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
Deep_Learning_y=[2.2085725821325958e-13, 1.3654974569625368e-13, 9.91332616070208e-14, 8.906203969934283e-14, 8.422805252015208e-14, 7.816373456698763e-14, 7.44374858616744e-14, 7.538795048148464e-14, 7.698882212652347e-14, 7.787514854325699e-14, 8.221693804470974e-14, 9.389515995450528e-14, 1.0521685233352507e-13, 1.1280774309491296e-13, 1.1865794909233227e-13, 1.3578268039906047e-13, 1.4848350556851558e-13, 1.6385200947205978e-13, 1.8272242803627697e-13]
plt.plot(Deep_Learning_x,Deep_Learning_y,label="HADAR(DLearning)",linewidth=2,color="red")


# Add axis labels and a title
plt.xlabel('E(TeV)')
plt.ylabel('E\'F(>E)(TeV cm-2s-1)')
# plt.title('Logarithmic x-axis')

plt.xlim(1e-3,1e3)
plt.ylim(1e-15,1e-10)
plt.xscale("log")
plt.yscale("log")
plt.legend()

# Display the plot
# plt.savefig("s.png",dpi=400)
plt.show()