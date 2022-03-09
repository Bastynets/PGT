import streamlit as st
from iapws import IAPWS97
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.write("Выполнено: Бастынец А.К. ФПэ-01-19")
st.write("Github: " + "https://github.com/Bastynets/PGT")
st.write("# Задание 1")
st.write("""Построить процесс расширения пара в турбине. Определение расходов пара на входе в турбину (G0) и в конденсатор (Gк). Получить зависимость КПД ПТУ от параметра заданного в таблице.""")
st.write("""# """)

st.write(" *Исходные данные:* ")

age = st.slider('Укажите максимальную границу t0', min_value = 500, max_value = 550, step = 1)
age = age + 0.001 # Посмотреть надобность

Ne = 212e6
p0 = 12.5e6

t0 = list(np.arange(500, age, 1.0))
T0 = [t+273.15 for t in t0]

ppp = 2.46e6
T0_min = float(T0[0])
T0_max = float(T0[-1])
tpp = 544
Tpp = tpp+273.15
pk = 4.2e3
tpv = 233
Tpv = tpv+273.15
delta_p_0 = 0.05*p0
delta_p_pp = 0.08*ppp
delta_p = 0.03*ppp

z = 8





st.write(""" P0 = """ + str(p0*10**(-6)) + """ МПа""")
st.write(""" t0 = """ + str(T0_min-273.15) + " - " + str(T0_max-273.15) + """ C""")
st.write(""" Pпп = """ + str(ppp*10**(-6)) + """ МПа""")
st.write(""" tпп = """ + str(tpp) + """ C """)
st.write(""" Pк = """ + str(pk*10**(-3)) + """ кПа """)
st.write(""" tпв = """ + str(tpv) + """ C """)
st.write(""" Nэ = """ + str(Ne*10**(-6)) + """ МВт """)
st.write(""" Z = """ + str(z) + """ шт """)



st.write("""# """)
st.write(" *Решение:* ")


def Calculate_eta_G0_Gk(N_e, p_0, T_0, p_pp, T_pp, p_k, T_pv):

    point_0 = IAPWS97(P=p_0*10**(-6), T=T_0)
    s_0 = point_0.s
    h_0 = point_0.h
    v_0 = point_0.v
    p_0_ = p_0-0.05*p_0
    point_p_0_ = IAPWS97(P=p_0_*10**(-6), h=h_0)
    t_0_ = point_p_0_.T-273.15
    s_0_ = point_p_0_.s
    v_0_ = point_p_0_.v


    p_1t = p_pp+0.1*p_pp
    point_1t = IAPWS97(P=p_1t*10**(-6), s=s_0)
    t_1t = point_1t.T-273.15
    h_1t = point_1t.h
    v_1t = point_1t.v


    point_pp = IAPWS97 (P=p_pp*10**(-6), T=T_pp)
    h_pp = point_pp.h
    s_pp = point_pp.s
    v_pp = point_pp.v

    H_0 = h_0-h_1t
    eta_oi = 0.85
    H_i_cvd = H_0*eta_oi

    h_1 = h_0 - H_i_cvd
    point_1 = IAPWS97(P = p_1t*10**(-6),h = h_1)
    s_1 = point_1.s
    T_1 = point_1.T
    v_1 = point_1.v
    p_pp_ = p_pp - 0.03*p_pp
    point_pp_ = IAPWS97(P=p_pp_*10**(-6),h = h_pp)
    s_pp_ = point_pp_.s
    v_pp_ = point_pp_.v
    point_kt = IAPWS97(P = p_k*10**(-6),s = s_pp)
    T_kt = point_kt.T
    h_kt = point_kt.h
    v_kt = point_kt.v
    s_kt = s_pp
    H_0_csdcnd = h_pp-h_kt
    eta_oi = 0.85
    H_i_csdcnd = H_0_csdcnd*eta_oi
    h_k = h_pp - H_i_csdcnd
    point_k = IAPWS97(P = p_k*10**(-6), h = h_k)
    T_k = point_k.T
    s_k = point_k.s
    v_k = point_k.v
    point_k_v = IAPWS97(P = p_k*10**(-6),x=0)
    h_k_v = point_k_v.h
    s_k_v = point_k_v.s
    eta_oiI = (h_1-h_0)/(h_1t-h_0)
    p_pv = 1.4*p_0
    point_pv = IAPWS97(P = p_pv*10**(-6),T=T_pv)
    h_pv = point_pv.h
    s_pv = point_pv.s
    ksi_pp_oo = 1-(1-(T_k*(s_pp-s_k_v))/((h_0-h_1t)+(h_pp-h_k_v)))/(1-(T_k*(s_pp-s_pv))/((h_0-h_1t)+(h_pp-h_pv)))
    #T_0_= IAPWS97(P = p_pv*10**(-6),x = 0).T
    T_0_ = 374.2+273.15
    T_ = (point_pv.T - point_k.T) / (T_0_ - point_k.T)
    if T_ <= 0.636364:
        ksi1 = -1.53*T_**2+2.1894*T_+0.0048
    elif 0.636364<T_<=0.736364:
        ksi1 = -1.3855*T_**2+2.0774*T_+0.0321
    elif 0.736364<T_<=0.863636:
        ksi1 = -2.6535*T_**2+4.2556*T_-0.8569


    if T_ <= 0.631818:
        ksi2 = -1.7131*T_**2+2.3617*T_-0.0142
    elif 0.631818<T_<=0.718182:
        ksi2 = -2.5821*T_**2+3.689*T_-0.4825
    elif 0.718182<T_<=0.827273:
        ksi2 = -1.9864*T_**2+3.138*T_-0.3626
    elif 0.827273<T_<=0.936364:
        ksi2 = -2.0619*T_**2+3.3818*T_-0.4814

    ksi = (ksi1+ksi2)/2
    ksi_r_pp = ksi*ksi_pp_oo
    eta_ir = (H_i_cvd+H_i_csdcnd)/(H_i_cvd+(h_pp-h_k_v))*1/(1-ksi_r_pp)
    H_i = eta_ir*((h_0-h_pv)+(h_pp-h_1))
    eta_m = 0.994
    eta_eg = 0.99
    G_0 = N_e/(H_i*eta_m*eta_eg*(10**3))
    G_k = N_e/((h_k-h_k_v)*eta_m*eta_eg*(10**3))*(1/eta_ir-1)

    return eta_ir, G_0, G_k


eta, G0, Gk = [], [], []
for t in T0:
    eta_ = Calculate_eta_G0_Gk(N_e=Ne, p_0=p0, T_0=t, p_pp=ppp, T_pp=Tpp, p_k=pk, T_pv=Tpv)
    eta.append(eta_[0])
    G0.append(eta_[1])
    Gk.append(eta_[2])





t0_f = [float(x) - 273.15 for x in T0]
eta_f = [float(x) * 100 for x in eta]

st.write(""" Максимальное КПД = """ + str('{:.4}'.format(float(eta_f[-1]))) + """ %""")
st.write(""" Расход пара на входе в турбину (G0) при макс. КПД = """ + str('{:.5}'.format(float(G0[-1]))) + """ кг/с""")
st.write(""" Расход пара на входе в конденсатор (Gк) при макс. КПД = """ + str('{:.5}'.format(float(Gk[-1]))) + """ кг/с""")
st.write("""# """)
st.write(" Табл. Зависимость КПД от t0 ")

t0_eta=pd.DataFrame({"t0, C": (t0_f),
                   "eta, %": (eta_f),
                   "G_0, кг/с": (G0),
                   "G_k, кг/с": (Gk)
                   })
st.dataframe(t0_eta)


st.write("""# """)

t0__eta= plt.figure()

plt.plot(t0_f, eta_f)
plt.plot(t0_f, eta_f, 'ro')
plt.title("Зависимость КПД от начальной температуры")
plt.xlabel("t0, C")
plt.ylabel("КПД, %")
plt.grid()

st.pyplot(t0__eta)







st.title(""" """)


fighs = plt.figure()

point_0 = IAPWS97(P=p0*1e-6, T=T0_max)
p_0_d = p0 - delta_p_0
point_0_d = IAPWS97(P=p_0_d*1e-6, h=point_0.h)
p_1t = ppp + delta_p_pp
point_1t = IAPWS97(P=p_1t*10**(-6), s=point_0.s)
H_01 = point_0.h - point_1t.h
kpd_oi = 0.85
H_i_cvd = H_01 * kpd_oi
h_1 = point_0.h - H_i_cvd
point_1 = IAPWS97(P=p_1t*1e-6, h=h_1)
point_pp = IAPWS97(P=ppp*1e-6, T=Tpp)
p_pp_d = ppp - delta_p_pp
point_pp_d = IAPWS97(P=p_pp_d*1e-6, h=point_pp.h)
point_kt = IAPWS97(P=pk*1e-6, s=point_pp.s)
H_02 = point_pp.h - point_kt.h
kpd_oi = 0.85
H_i_csd_cnd = H_02 * kpd_oi
h_k = point_pp.h - H_i_csd_cnd
point_k = IAPWS97(P=pk*1e-6, h=h_k)

s_0 = [point_0.s-0.05,point_0.s,point_0.s+0.05]
h_0 = [IAPWS97(P = p0*1e-6,s = s_).h for s_ in s_0]
s_1 = [point_0.s-0.05,point_0.s,point_0.s+0.18]
h_1 = [IAPWS97(P=p_1t*1e-6, s = s_).h for s_ in s_1]
s_0_d = [point_0_d.s-0.05, point_0_d.s, point_0_d.s+0.05]
h_0_d = h_0
s_pp = [point_pp.s-0.05,point_pp.s,point_pp.s+0.05]
h_pp = [IAPWS97(P=ppp*1e-6, s=s_).h for s_ in s_pp]
s_k = [point_pp.s-0.05,point_pp.s,point_pp.s+0.8]
h_k = [IAPWS97(P=pk*1e-6, s=s_).h for s_ in s_k]
s_pp_d = [point_pp_d.s-0.05,point_pp_d.s,point_pp_d.s+0.05]
h_pp_d = h_pp

plt.plot([point_0.s,point_0.s,point_0_d.s,point_1.s],[point_1t.h,point_0.h,point_0.h,point_1.h],'-or')
plt.plot([point_pp.s,point_pp.s,point_pp_d.s,point_k.s],[point_kt.h,point_pp.h,point_pp.h,point_k.h],'-or')
plt.plot(s_0,h_0)
plt.plot(s_1,h_1)
plt.plot(s_0_d,h_0_d)
plt.plot(s_pp,h_pp)
plt.plot(s_k,h_k)
plt.plot(s_pp_d,h_pp_d)

for x, y, ind in zip([point_pp.s, point_k.s], [point_pp.h, point_k.h], ['{пп}', '{к}']):
    plt.text(x-0.45, y+40, '$h_' + ind + ' = %.2f $'%y)
for x, y, ind in zip([point_kt.s, point_pp_d.s], [point_kt.h, point_pp_d.h], ['{кт}', '{ппд}']):
    plt.text(x+0.03, y+40, '$h_' + ind + ' = %.2f $'%y)

for x, y, ind in zip ([point_0.s, point_1.s], [point_0.h, point_1.h], ['{0}', '{1}']):
    plt.text(x-0.01, y+120, '$h_' + ind + ' = %.2f $'%y)

for x, y, ind in zip([point_1t.s, point_0_d.s], [point_1t.h, point_0_d.h], ['{1т}', '{0д}']):
    plt.text(x+0.03, y-60, '$h_' + ind + ' = %.2f $'%y)


    plt.title("h - s диаграмма")
    plt.xlabel("s, кДж/(кг*С)")
    plt.ylabel("h, кДж/кг")
    plt.grid(True)


st.pyplot(fighs)

