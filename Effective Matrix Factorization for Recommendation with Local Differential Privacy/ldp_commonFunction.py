import common_function as cf
import iterationRelated as iR
import numpy as np
import math


def readfile(index):
    fr = open(index + 'ratings.txt')
    readLines = fr.readlines()
    s = np.alen(readLines)
    n = []
    m = []
    r = []
    # print(readLines)
    k = 1
    for line in readLines:
        # print(line)
        if k == 1:
            k += 1
            continue
        ListFormLine = line.strip()
        ListFormLine = ListFormLine.split('\t')
        n.append(int(ListFormLine[0]) - 1)
        m.append(int(ListFormLine[1]) - 1)
        r.append(float(ListFormLine[2]))
    fr.close()
    np.save(index + 'n', n)
    np.save(index + 'm', m)
    np.save(index + 'r', r)
    print('第一步：已获得m，n的值。', s)
    return True


def adjustTheDataset(index):
    n = np.load(index + 'n.npy')
    m = np.load(index + 'm.npy')
    r = np.load(index + 'r.npy')
    ratings_number = len(n)
    r_arr = np.zeros((3, ratings_number))
    b = []
    for i in range(ratings_number):
        if n[i] not in b:
            b.append(n[i])
    sum_number = 0
    l_b = len(b)
    n_nb = l_b
    for i in range(ratings_number):
        for j in range(l_b):
            if n[i] == b[j]:
                n[i] = j
                sum_number += 1
                break

    b = []
    for i in range(ratings_number):
        if m[i] not in b:
            b.append(m[i])
    sum_number = 0
    l_b = len(b)
    m_mb = l_b
    for i in range(ratings_number):
        for j in range(l_b):
            if m[i] == b[j]:
                m[i] = j
                sum_number += 1
                break
    print(sum_number)
    r_arr[0] = n
    r_arr[1] = m
    r_arr[2] = r
    sst = r_arr.T[np.lexsort(r_arr[::-1, :])].T
    n = sst[0]
    m = sst[1]
    r = sst[2]
    np.save(index + 'n', n)
    np.save(index + 'm', m)
    np.save(index + 'r', r)
    print(n_nb, m_mb)
    print(n[:10], m[:10])
    return n, m, r, n_nb, m_mb


def get_flag(r, c, ratings, m, n, sum11, eta, index):
    i = 0
    # click = 0
    sum_ratings = len(r)
    flag = []
    flag_v = []
    user_item = []
    other_item = []
    for rs_1 in range(m):
        other_item.append(rs_1)
    other_item_i = other_item[:]
    for rs in range(sum_ratings):
        flag.append(0)
    for click in range(sum_ratings):
        row = r[click]
        if row == i:
            user_item.append(c[click])
            other_item_i.remove(c[click])
        else:
            p = np.random.random()
            if p > (math.exp(eta)/(math.exp(eta) + 1)):
                s = len(other_item_i)
                l = np.random.randint(0, s-1)
                flag_v.append(other_item_i[l])
            else:
                s = len(user_item)
                l = np.random.randint(0, s-1)
                flag_v.append(other_item_i[l])
                flag[click-s+l] = 1
    np.save(index + f'flag{sum11}', flag)
    np.save(index + f'flag_v{sum11}', flag_v)


def get_frequency(m, flag_v, index, sum11):
    lens = len(flag_v)
    items = []
    fr = []
    frency = []
    for i in range(m):
        fr.append(0)
    print('1')
    for i in range(lens):
        item_v = int(flag_v[i])
        fr[item_v] += 1
    print('get items')
    for j in range(lens):
        item_v = int(flag_v[j])
        frency.append(fr[item_v])
    print('get fr')
    np.save(index + f'fr{sum11}', frency)


def machine_learning(r, c, ratings, m, n, d, sum11, U, V, k, index):
    sum_ratings = len(ratings)
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    # k = 1
    for it in range(k):
        rt_u = 1 / (it + 1)
        rt_v = 1 / (it + 1)
        # rt_v = 1 / (it + 1) / (k ** 2)
        # 验证实验
        # rt = 1 / (it+1) / (k ** 2)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dV = np.zeros((m, d))
        sum1 = 0
        i = 0
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            dV[j] += -2 * U[i] * (ratings[sum1] - T)
            dU[i] += (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    i += 1
            sum1 += 1
        dV = dV / n
        dU /= n
        U = U - rt_u * (dU + 2 * lu * U)
        V = V - rt_v * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(index + f'U_0{sum11}', U)
    np.save(index + f'V_0{sum11}', V)


def Xiao(m, n, d, sum11, U, V, eta, k, index):
    r = np.load(index + 'n.npy')
    c = np.load(index + 'm.npy')
    ratings = np.load(index + 'r.npy')
    D = np.load(index + 'D.npy')
    D3 = np.load(index + 'D3.npy')
    sum_ratings = len(ratings)
    index = index + "xiao\\"
    # m = 131262
    # n = 138493
    # d = 15
    q = 2700
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    # k = 10
    # sum = 20000263
    # eta = 0.1
    t_sum = q * d * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_sum_1 = math.exp(eta/k)
    for it in range(k):
        rt_u = 1 / (it + 1)
        rt_v = 1 / (it + 1) / (k ** 2)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((q, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            # if j >= 0:
            dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum1]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    # x = np.dot(D, dV)
                    s = np.random.randint(0, q)
                    ls = np.random.randint(0, d)
                    # t = x[s][ls]
                    t = np.dot(D[s], (dV.T)[ls])
                    if t > 1:
                        t = 1
                    elif t < -1:
                        t = -1
                    T = (t * (t_sum_1 - 1) + t_sum_1 + 1) / (2 * (t_sum_1 + 1))
                    random_t = np.random.random()
                    if random_t <= T:
                        dB[s][ls] = t_sum
                    else:
                        dB[s][ls] = -t_sum
                    dV = np.zeros((m, d))
                    i += 1
            else:
                # x = np.dot(D, dV)
                s = np.random.randint(0, q)
                ls = np.random.randint(0, d)
                # t = x[s][ls]
                t = np.dot(D[s], (dV.T)[ls])
                if t > 1:
                    t = 1
                elif t < -1:
                    t = -1
                T = (t * (t_sum_1 - 1) + t_sum_1 + 1) / (2 * (t_sum_1 + 1))
                random_t = np.random.random()
                if random_t <= T:
                    dB[s][ls] = t_sum
                else:
                    dB[s][ls] = -t_sum
            sum1 += 1
        dB = dB / n
        dU /= n
        dV = np.dot(D3, dB)
        # dV_t = np.dot(D, dV)
        # dV = np.dot(D3, dV_t)
        U = U - rt_u * (dU + 2 * lu * U)
        V = V - rt_v * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(index + f'U_2{sum11}', U)
    np.save(index + f'V_2{sum11}', V)


def Hua(r, c, ratings, lu, lv, sub_index, U, V, eta, m, n, d, index, k):
    # lu = 10 ** -8
    # lv = 10 ** -8
    # k = 10
    """
    U = np.random.rand(n, d)
    V = np.random.rand(m, d)
    np.save(index + 'u_4', U)
    np.save(index + 'v_4', V)
    U = np.load(index + 'u_4.npy')
    V = np.load(index + 'v_4.npy')
    """
    sum_ratings = len(r)
    index = index + "hua\\"
    # k = 1
    # rt = (2 ** -5)
    for it in range(k):
        # rt = 1 / (it + 1)
        rt_u = 1 / (it + 1)
        rt_v = 1 / (it + 1)
        # rt_u = (2 ** -5)
        # rt_v = rt_u
        print(str(it) + ' iterations！')
        cf.show_time()
        dU = np.zeros((n, d))
        dV = np.zeros((m, d))
        sum1 = 0
        i = 0
        while sum1 < sum_ratings:
            j = int(c[sum1])
            T = np.dot(U[i], V[j].T)
            v_grant = -2 * U[i] * (ratings[sum1] - T)
            dV[j] += v_grant
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            click = sum1 + 1
            if click < sum_ratings:
                try:
                    test = r[click]
                except Exception as er:
                    print(er, test)
                    break
                if r[click] != i:
                    # dV = np.zeros((m, d))
                    i += 1
            sum1 += 1
        dV = dV + np.random.laplace(0, 10 * (d ** 0.5) / eta, (m, d)) * 200 * (4/3)
        dV = dV / n
        # dV = dV + np.random.laplace(0, 2 / eta, (m, d))
        dU /= n
        U = U - rt_u * (dU + 2 * lu * U)
        V = V - rt_v * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sub_index} Hua')
    np.save(index + f'U_4{sub_index}', U)
    np.save(index + f'V_4{sub_index}', V)


def Private_GD_DR(m, n, d, sum11, U, V, eta, k, c_flag, l_array, r_q, index):
    r = np.load(index + 'n.npy')
    c = np.load(index + 'm.npy')
    ratings = np.load(index + 'r.npy')
    D = np.load(index + 'Dl1.npy')
    D3 = np.load(index + 'Dl3.npy')
    n, l_items = np.shape(l_array)
    q = r_q
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    sum_ratings = len(r)
    t_sum = q * d * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_s = math.exp(eta/k)
    # print(l_array[:10][:10], c_flag[:10])
    for it in range(k):
        # rt = 1 / (it + 1) / (k ** 2)
        rt_u = 1 / (it + 1)
        rt_v = 1 / (it + 1) / k
        # rt = 1 / (it + 1) / k
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        dl = np.zeros((l_items, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            if c_flag[sum1] == 1:
                dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    for o in range(l_items):
                        tts = int(l_array[i][o])
                        dl[o] = dV[tts]
                    x = np.dot(D, dl)
                    x_i = np.zeros((q, d))
                    s = np.random.randint(0, q)
                    ls = np.random.randint(0, d)
                    t = x[s][ls]
                    if t > 1:
                        t = 1
                    elif t < -1:
                        t = -1
                    T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                    random_t = np.random.random()
                    if random_t <= T:
                        x_i[s][ls] = t_sum
                    else:
                        x_i[s][ls] = -t_sum
                    x_z = np.dot(D3, x_i)
                    for o in range(l_items):
                        tts = int(l_array[i][o])
                        dB[tts] += x_z[o]
                    dV = np.zeros((m, d))
                    i += 1
            else:
                for o in range(l_items):
                    tts = int(l_array[i][o])
                    dl[o] = dV[tts]
                x = np.dot(D, dl)
                x_i = np.zeros((q, d))
                s = np.random.randint(0, q)
                ls = np.random.randint(0, d)
                t = x[s][ls]
                if t > 1:
                    t = 1
                elif t < -1:
                    t = -1
                T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                random_t = np.random.random()
                if random_t <= T:
                    x_i[s][ls] = t_sum
                else:
                    x_i[s][ls] = -t_sum
                x_z = np.dot(D3, x_i)
                for o in range(l_items):
                    tts = int(l_array[i][o])
                    dB[tts] += x_z[o]
                # dV = np.zeros((m, d))
            sum1 += 1
        dV = dB / n
        dU /= n
        U = U - rt_u * (dU + 2 * lu * U)
        V = V - rt_v * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(index + f'U_6{sum11}', U)
    np.save(index + f'V_6{sum11}', V)


def Private_GD_DR_1(m, n, d, sum11, U, V, eta, k, c_flag, l_array, index):
    r = np.load(index + 'n.npy')
    c = np.load(index + 'm.npy')
    ratings = np.load(index + 'r.npy')
    # D = np.load('E:\\movielens\\Dl1.npy')
    # D3 = np.load('E:\\movielens\\Dl3.npy')
    n, l_items = np.shape(l_array)
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    sum_ratings = len(r)
    t_sum = l_items * d * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_s = math.exp(eta/k)
    # print(l_array[:10][:10], c_flag[:10])
    for it in range(k):
        # rt = 1 / (it + 1) / (k ** 2)
        rt_u = 1 / (it + 1)
        rt_v = 1 / (it + 1)
        # rt = 1 / (it + 1) / k
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        dl = np.zeros((l_items, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            if c_flag[sum1] == 1:
                dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    for o in range(l_items):
                        tts = int(l_array[i][o])
                        dl[o] = dV[tts]
                    s = np.random.randint(0, l_items)
                    v_s = l_array[i][s]
                    ls = np.random.randint(0, d)
                    t = dl[s][ls]
                    if t > 1:
                        t = 1
                    elif t < -1:
                        t = -1
                    T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                    random_t = np.random.random()
                    if random_t <= T:
                        dB[v_s][ls] += t_sum
                    else:
                        dB[v_s][ls] = -t_sum
                    dV = np.zeros((m, d))
                    i += 1
            else:
                for o in range(l_items):
                    tts = int(l_array[i][o])
                    dl[o] = dV[tts]
                s = np.random.randint(0, l_items)
                v_s = l_array[i][s]
                ls = np.random.randint(0, d)
                t = dl[s][ls]
                if t > 1:
                    t = 1
                elif t < -1:
                    t = -1
                T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                random_t = np.random.random()
                if random_t <= T:
                    dB[v_s][ls] += t_sum
                else:
                    dB[v_s][ls] = -t_sum
            sum1 += 1
        dV = dB / n
        dU /= n
        U = U - rt_u * (dU + 2 * lu * U)
        V = V - rt_v * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(index + f'U_7{sum11}', U)
    np.save(index + f'V_7{sum11}', V)


def zh_select_GD_DR(n, m, r, c, ratings, d, sum11, U, V, eta, k, index, frency, flag, flag_v, delt):
    sum_ratings = len(ratings)
    lu = 10 ** -8
    lv = 10 ** -8
    factor = eta * math.sqrt(n) / ((delt * d * k) * math.sqrt(2*d))
    # k = 10
    # k = 1
    for it in range(k):
        rt_u = 1 / (it + 1)
        rt_v = 1 / (it + 1) * factor
        # rt_v = 1 / (it + 1) / (k ** 2)
        # 验证实验
        # rt = 1 / (it+1) / (k ** 2)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dV = np.zeros((m, d))
        sum1 = 0
        i = 0
        f = 0
        # factor = eta / ((delt * d * k) * math.sqrt(2*d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            if flag[sum1] == 1 and f == 0:
                # factor_temp = factor / math.sqrt(frency[i])
                dV[j] += np.random.laplace(0, delt*d*k/eta, d) - 2 * U[i] * (ratings[sum1] - T)
                # if factor_temp < 1:
                #     dV[j] += (np.random.laplace(0, 2*5*d*k/eta, d) - 2 * U[i] * (ratings[sum1] - T)) * factor_temp
                # else:
                #     dV[j] += -2 * U[i] * (ratings[sum1] - T)
                f = 1
            dU[i] += (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    ss = int(flag_v[i])
                    if f == 0:
                        # factor_temp = factor / math.sqrt(frency[i])
                        dV[ss] += np.random.laplace(0, delt*d*k/eta, d)
                        # if factor_temp < 1:
                        #     dV[ss] += (np.random.laplace(0, 2*5*d*k/eta, d)) * factor_temp
                        # else:
                        #     dV[ss] += np.random.laplace(0, 2*5*d*k/eta, d)
                    # 可以考虑使用Nuyen提出的算法，单个加噪，另外可以尝试看
                    f = 0
                    i += 1
            sum1 += 1
        dV = dV / n
        dU /= n
        U = U - rt_u * (dU + 2 * lu * U)
        V = V - rt_v * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(index + f'U_7{sum11}', U)
    np.save(index + f'V_7{sum11}', V)


def get_rmse(r, c, ratings, U, V):
    n = len(r)
    rmse = 0
    i = 0
    for m in range(n):
        i = int(r[m])
        j = int(c[m])
        rmse += ((ratings[m] - np.dot(U[i], V[j].T)) ** 2)
    rmse = (rmse / n) ** 0.5
    print(rmse)
    return rmse


def get_vmax(r, c, ratings, index, ss):
    v = np.load(index + 'v_00.npy')
    vmax = []
    rmse = []
    index = index + "hua\\"
    for i in range(2):
        j = i
        # ML_LDP_prove(m, n, sum_ratings, d, i, u, v, eta_list[i], k)
        # ml_LDP(m, n, d, j, u, v, eta_list[j])
        # ml_LDP_hau(i, u, v, eta_list[i], m, n, sum_ratings, d)
        # ml_LDP_NJ(i, u, v, eta_list[i], m, n, sum_ratings, d)
        v1 = np.load(index + f'V_{ss}{j}.npy')
        u1 = np.load(index + f'U_{ss}{j}.npy')
        rmse_u = get_rmse(r, c, ratings, u1, v1)
        rmse.append(rmse_u)
        temp = np.linalg.norm(v-v1, np.inf)
        vmax.append(temp)
        print('I have finish the epslion of ', ss, temp)
    print(vmax)
    print(rmse)


def get_fScore(index, table, n):
    u = np.load(index + 'u_00.npy')
    v = np.load(index + 'v_00.npy')
    user_top_items = getTopK(u, v, n, index)
    user_top_items = np.load(index + 'user_top_items.npy')

    f_score_list = []
    for i in range():
        u = np.load(index + f'U_{table}{i}.npy')
        v = np.load(index + f'V_{table}{i}.npy')
        # get_percent(r, c, ratings, u, v)
        f_score_number = f_score(n, user_top_items, u, v)
        f_score_list.append(f_score_number)
    print(f_score_list)


def get_percent(r, c, ratings, U, V):
    error_list = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    sum_ratings = len(r)
    sum1 = 0
    i = 0
    while sum1 < sum_ratings:
        j = int(c[sum1])
        T = np.dot(U[i], V[j].T)
        if abs(ratings[sum1]-T) < 0.5:
            error_list[0] += 1
        elif abs(ratings[sum1]-T) < 1:
            error_list[1] += 1
        elif abs(ratings[sum1]-T) < 1.5:
            error_list[2] += 1
        elif abs(ratings[sum1]-T) < 2:
            error_list[3] += 1
        elif abs(ratings[sum1]-T) < 2.5:
            error_list[4] += 1
        elif abs(ratings[sum1]-T) < 3:
            error_list[5] += 1
        elif abs(ratings[sum1]-T) < 3.5:
            error_list[6] += 1
        elif abs(ratings[sum1]-T) < 4:
            error_list[7] += 1
        elif abs(ratings[sum1]-T) < 4.5:
            error_list[8] += 1
        elif abs(ratings[sum1]-T) < 5:
            error_list[9] += 1
        elif abs(ratings[sum1]-T) < 5.5:
            error_list[10] += 1
        elif abs(ratings[sum1]-T) < 6:
            error_list[11] += 1
        elif abs(ratings[sum1]-T) < 6.5:
            error_list[12] += 1
        elif abs(ratings[sum1]-T) < 7:
            error_list[13] += 1
        elif abs(ratings[sum1]-T) < 7.5:
            error_list[14] += 1
        elif abs(ratings[sum1]-T) < 8:
            error_list[15] += 1
        elif abs(ratings[sum1]-T) < 8.5:
            error_list[16] += 1
        elif abs(ratings[sum1]-T) < 9:
            error_list[17] += 1
        elif abs(ratings[sum1]-T) < 9.5:
            error_list[18] += 1
        elif abs(ratings[sum1]-T) < 10:
            error_list[19] += 1
        sum1 += 1
        if sum1 < sum_ratings:
            try:
                test = r[sum1]
            except Exception as er:
                print(er, test)
                break
            if r[sum1] != i:
                i += 1
    for i in range(19):
        error_list[i+1] += error_list[i]
    for i in range(20):
        error_list[i] = error_list[i] / sum_ratings
    print(error_list)


def getTopK(u, v, n, index):
    cf.show_time()
    user_top_items = np.zeros((n, 10))
    for i in range(n):
        user_score_0 = np.dot(u[i], v.T)
        list_0 = np.argsort(-user_score_0)
        user_top_items[i] = list_0[:10]
    np.save(index + 'user_top_items', user_top_items)
    cf.show_time()
    return user_top_items


def f_score(user_len, user_top_items, u1, v1):
    cf.show_time()
    f_score_sum = 0
    for i in range(user_len):
        user_score_1 = np.dot(u1[i], v1.T)
        list_1 = np.argsort(-user_score_1)
        temp_count = 0
        for j in range(10):
            if list_1[j] in user_top_items[i]:
                temp_count += 1
        f_score_sum += temp_count / 10
    f_score_final = f_score_sum / user_len
    print(f_score_final)
    cf.show_time()
    return f_score_final


def LDP_PD(m, n, d, u, v, k, l_i, eta_list, l_q, index):
    # use ml_LDP_part algrithm
    # produce_D_D3(l_i, l_q)
    for j in range(6):
        i = j
        eta = eta_list[i] / 2
        # s_items(m, n, d, i, u, v, eta, k, l_i)
        print('get s_items')
        l_array = np.load(index + f'l_array{i}.npy')
        # get_flag(m, n, i, l_array)
        print('get_flag')
        c_flag = np.load(index + f'c_flag{i}.npy')
        # ML_LDP_part(m, n, d, i, u, v, eta, k, c_flag, l_array, l_q)
        Private_GD_DR(m, n, d, i, u, v, eta, k, c_flag, l_array, l_q, index)
        print('get the granted')


def LibimSeTi():
    eta_list = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    # n, m = find_n_m()
    # d_limbi(m, n)
    # n, m = find_n_m()
    index = 'E:\\limbi\\'
    n = 135359
    m = 26509
    d = 20
    lu = 10 ** -8
    lv = 10 ** -8
    q = 2700
    k = 10
    l_i = 50
    l_q = 5
    # second step
    # u = np.random.rand(n, d)
    # v = np.random.rand(m, d)
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
    ratings = np.load('E:\\limbi\\r.npy')
    u = np.load('E:\\limbi\\u.npy')
    v = np.load('E:\\limbi\\v.npy')
    # machine_learning(r, c, ratings, m, n, d, 0, u, v, k, index)
    # u = np.load('E:\\limbi\\u_00.npy')
    # v = np.load('E:\\limbi\\v_00.npy')
    # user_top_items = np.load(index + 'user_top_items.npy')
    """
    for i in range(6):
        j = i
        Hua(r, c, ratings, lu, lv, j, u, v, eta_list[j], m, n, d, index, k)
    LDP_PD(m, n, d, u, v, k, l_i, eta_list, l_q, index)
    # user_top_items = np.load(index + 'user_top_items.npy')
    ss_list = [4]
    # for i in range(2):
    get_vmax(r, c, ratings, index, ss_list[0])
    """
    user_top_items = np.load(index + "user_top_items.npy")
    v_list = [0, 2]
    U = np.load(index+"iR\\U_06.npy")
    V = np.load(index + "iR\\V_06.npy")
    get_percent(r, c, ratings, U, V)
    # get_vmax(r, c, ratings, index, 4)
    # get_vmax(r, c, ratings, index, 0)
    # index = index + "iR\\"
    for i in range(2):
        iR.get_granted(r, c, ratings, m, n, d, i+6, u, v, k, index, eta_list[i], 1, 0)
        iR.get_granted(r, c, ratings, m, n, d, i+6, u, v, k, index, eta_list[i], 2*k/eta_list[i], 2)
        Xiao(m, n, d, i, u, v, eta_list[i], k, index)
        Hua(r, c, ratings, lu, lv, i, u, v, eta_list[i], m, n, d, index, k)
    """
    for i in range(2):
        f_score_final_list = []
        for j in range(6):
            # ML_LDP_prove(m, n, sum_ratings, d, i, u, v, eta_list[i], k)
            # ml_LDP(m, n, d, j, u, v, eta_list[j])
            # ml_LDP_hau(i, u, v, eta_list[i], m, n, sum_ratings, d)
            # ml_LDP_NJ(i, u, v, eta_list[i], m, n, sum_ratings, d)
            v1 = np.load(index + f'V_{v_list[i]}{j}.npy')
            u1 = np.load(index + f'U_{v_list[i]}{j}.npy')
            # get_percent(r, c, ratings, u1, v1)
            f_score_final_list.append(f_score(n, user_top_items, u1, v1))
        print(f_score_final_list)
    """


def MovieLens():
    q = 2700
    m = 26744
    n = 138493
    d = 15
    l_i = 50
    l_q = 5
    k = 10
    lu = 10 ** -8
    lv = 10 ** -8
    eta_list = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    index = 'E:\\movielens\\'
    r = np.load(index + 'n.npy')
    c = np.load(index + 'm.npy')
    # c = np.load('E:\\movielens\column_list.npy')
    ratings = np.load(index + 'r.npy')
    u = np.load(index + 'u.npy')
    v = np.load(index + 'v.npy')
    ss_list = [4]
    delt = 10
    # get_vmax(r, c, ratings, index, ss_list[0])
    k_list = [1, 2, 3, 4, 5, 10, 20, 50]
    """
    for i in range(5):
        j = i + 1
        eta = eta_list[i]/2
        flag = np.load(index + f'flag{j}.npy')
        flag_v = np.load(index + f'flag_v{j}.npy')
        # get_flag(r, c, ratings, m, n, j, eta, index)
        frency = np.load(index + f'fr{j}.npy')
        # get_frequency(m, flag_v, index, j)
        zh_select_GD_DR(n, m, r, c, ratings, d, j+6, u, v, eta, k, index, frency, flag, flag_v, delt)
        # Hua(r, c, ratings, lu, lv, j+6, u, v, eta_list[1], m, n, d, index, k_list[j])
        # machine_learning(r, c, ratings, m, n, d, 1, u, v, k, index)
    # LDP_PD(m, n, d, u, v, k, l_i, eta_list, l_q, index)
    # user_top_items = np.load(index + 'user_top_items.npy')
    ss_list = [7]
    """

    # Xiao(m, n, d, 2, u, v, 0.00625, k, index)
    # for i in range(2):
        # Hua(r, c, ratings, lu, lv, i, u, v, eta_list[i], m, n, d, index, k)

    
    user_top_items = np.load(index + "user_top_items.npy")
    v_list = [0, 2]
    index = index + "xiao//"

    for i in range(2):
        f_score_final_list = []
        for j in range(2):
            # ML_LDP_prove(m, n, sum_ratings, d, i, u, v, eta_list[i], k)
            # ml_LDP(m, n, d, j, u, v, eta_list[j])
            # ml_LDP_hau(i, u, v, eta_list[i], m, n, sum_ratings, d)
            # ml_LDP_NJ(i, u, v, eta_list[i], m, n, sum_ratings, d)
            v1 = np.load(index + f'V_{v_list[i+1]}{j}.npy')
            u1 = np.load(index + f'U_{v_list[i+1]}{j}.npy')
            # get_percent(r, c, ratings, u1, v1)
            f_score_final_list.append(f_score(n, user_top_items, u1, v1))
            # get_rmse(r, c, ratings, u1, v1)
            # f_score_final_list.append(get_rmse(r, c, ratings, u1, v1))
        print(f_score_final_list)

    index = index + "iR\\"
    U = np.load(index+"U_06.npy")
    V = np.load(index+"V_06.npy")
    get_percent(r, c, ratings, U, V)
    v1 = np.load(index + 'v_00.npy')
    index = index + 'hua\\'
    v2 = np.load(index + 'v_41.npy')
    u2 = np.load(index + 'u_41.npy')
    get_rmse(r, c, ratings, u2, v2)
    print(np.linalg.norm(v1-v2, np.inf))


def set_100k():
    index = 'E:\\100k\\'
    # readfile(index)
    # adjustTheDataset(index)
    n = 943
    m = 1682
    d = 50
    lu = 0.001
    lv = 0.001
    r = np.load(index + 'n.npy')
    c = np.load(index + 'm.npy')
    # c = np.load('E:\\movielens\column_list.npy')
    ratings = np.load(index + 'r.npy')
    u = np.random.rand(n, d)
    print(u[10])
    v = np.random.rand(m, d)
    print(v[10])

    eta = 0.1
    k_list = [20, 40, 60, 80, 100]
    for i in range(5):
        # Hua(r, c, ratings, lu, lv, i, u, v, eta, m, n, d, index, k_list[i])
        # get_vmax(r, c, ratings, index, 4)
        u = np.load(index + f'u_4{i}.npy')
        v = np.load(index + f'v_4{i}.npy')
        get_percent(r, c, ratings, u, v)


if __name__ == '__main__':
    cf.show_title()
    # MovieLens()
    LibimSeTi()
    # set_100k()
