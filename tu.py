# -*- coding: utf-8 -*-
"""
重生成专利图（图3-图11）：
1) 放大字体
2) 去掉“图几”字样
3) 去掉“权利要求/有益效果”措辞
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 220
rcParams['font.size'] = 16
rcParams['axes.titlesize'] = 22
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14

ROOT = r"e:\\Desktop\\储能变流器数据处理"
DIR_A = os.path.join(ROOT, 'patent_figures')
DIR_B = os.path.join(ROOT, 'patent_figures_v3')


def _box(ax, x, y, w, h, text, fs=20):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.02',
                       linewidth=2.8, facecolor='#f7f7f7', edgecolor='black')
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)


def _arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=34,
                        linewidth=2.4, color='black')
    ax.add_patch(a)


def regenerate_fig3():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.set_title('数据处理与特征构造管线示意图', pad=20, fontweight='bold')

    _box(ax, 0.04, 0.62, 0.18, 0.20, '原始运行数据\n(低频RMS)', fs=24)
    _box(ax, 0.30, 0.70, 0.20, 0.12, '清洗数据\n尖峰检测+中值滤波', fs=22)
    _box(ax, 0.30, 0.50, 0.20, 0.12, '趋势数据\n滑动平均+漂移/突变', fs=22)
    _box(ax, 0.30, 0.30, 0.20, 0.12, '高级特征数据\n小波去噪(可选)', fs=22)
    _box(ax, 0.56, 0.38, 0.20, 0.36, '特征向量 X(t)\n15+维(含派生特征)', fs=24)
    _box(ax, 0.80, 0.70, 0.16, 0.12, '分层诊断\n输出故障项', fs=22)
    _box(ax, 0.80, 0.50, 0.16, 0.12, '异常检测\n异常点+贡献', fs=22)
    _box(ax, 0.80, 0.30, 0.16, 0.12, '变换分析\n序分量/矢量', fs=22)
    _box(ax, 0.56, 0.08, 0.40, 0.14, '融合评分/健康度/维护建议\n并生成CSV、文本与PDF(含图表)', fs=22)

    _arrow(ax, 0.22, 0.72, 0.30, 0.76)
    _arrow(ax, 0.22, 0.72, 0.30, 0.56)
    _arrow(ax, 0.22, 0.72, 0.30, 0.36)

    _arrow(ax, 0.50, 0.76, 0.56, 0.66)
    _arrow(ax, 0.50, 0.56, 0.56, 0.56)
    _arrow(ax, 0.50, 0.36, 0.56, 0.46)

    _arrow(ax, 0.76, 0.66, 0.80, 0.76)
    _arrow(ax, 0.76, 0.56, 0.80, 0.56)
    _arrow(ax, 0.76, 0.46, 0.80, 0.36)

    _arrow(ax, 0.88, 0.70, 0.76, 0.15)
    _arrow(ax, 0.88, 0.50, 0.76, 0.15)
    _arrow(ax, 0.88, 0.30, 0.76, 0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_A, 'fig3_data_pipeline.png'))
    plt.close()


def regenerate_fig4():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.set_title('深度学习多步预测结构示意图', pad=20, fontweight='bold')

    _box(ax, 0.05, 0.62, 0.22, 0.22, '输入序列\n长度L=24(优选)\n特征维度=15', fs=24)
    _box(ax, 0.34, 0.70, 0.16, 0.14, '归一化\nMinMax', fs=24)
    _box(ax, 0.53, 0.70, 0.18, 0.14, '双向LSTM(64)\n+BN+Dropout', fs=22)
    _box(ax, 0.53, 0.47, 0.18, 0.14, '双向LSTM(32)\n+BN+Dropout', fs=22)
    _box(ax, 0.74, 0.58, 0.16, 0.22, 'Dense\n输出维度\nH×Targets', fs=22)
    _box(ax, 0.74, 0.24, 0.22, 0.20, '输出预测\nH=6(优选)\n目标: T_hot,T_cpu,η', fs=22)
    _box(ax, 0.05, 0.08, 0.60, 0.14, '风险评估\n阈值(优选): T_w=75°C, T_c=85°C, η_w=0.90\n输出: 风险等级/故障概率/到阈值时间', fs=21)

    _arrow(ax, 0.27, 0.73, 0.34, 0.77)
    _arrow(ax, 0.50, 0.77, 0.53, 0.77)
    _arrow(ax, 0.62, 0.70, 0.62, 0.61)
    _arrow(ax, 0.71, 0.77, 0.74, 0.69)
    _arrow(ax, 0.82, 0.58, 0.82, 0.44)
    _arrow(ax, 0.79, 0.44, 0.56, 0.15)

    ax.text(0.05, 0.01, '注: L与H可选调整; 采样间隔优选5分钟。', fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_A, 'fig4_lstm_structure.png'))
    plt.close()


def upscale_crop_top(path, scale=1.35, crop_ratio=0.06):
    if not os.path.exists(path):
        return
    im = Image.open(path)
    w, h = im.size
    crop_top = int(h * crop_ratio)
    im = im.crop((0, crop_top, w, h))
    nw, nh = int(im.size[0] * scale), int(im.size[1] * scale)
    im = im.resize((nw, nh), Image.LANCZOS)
    im.save(path)


def regenerate_v3_figures_5_11():
    np.random.seed(42)

    # fig5
    t = np.linspace(0, 24, 288)
    p = np.clip(250 + 180*np.sin(2*np.pi*(t-6)/24) + np.random.normal(0, 8, len(t)), 50, 500)
    tenv = 28 + 6*np.sin(2*np.pi*(t-14)/24) + np.random.normal(0, 0.3, len(t))
    eta = 0.98 - 0.02*(1-p/500)**2 - 0.015*(p/500)
    tl = tenv + p*(1-eta)*0.12
    tm = tl + np.random.normal(0, 1.2, len(t))
    d = 5.0
    rphy = tm - (tl + d)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios':[1.2,1.2,1]})
    fig.suptitle('动态电-热耦合模型温度计算验证', fontweight='bold')
    ax1 = axes[0]
    ax1.plot(t, p, color='#e74c3c', lw=1.8, label='有功功率 P (kW)')
    ax1.set_ylabel('有功功率 (kW)')
    ax1.grid(True, alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(t, tenv, '--', color='#3498db', lw=1.8, label='环境温度 Tenv (°C)')
    ax1b.set_ylabel('环境温度 (°C)', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1b.tick_params(axis='y', labelcolor='#3498db')
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')
    ax1.set_title('输入参数：实时有功功率 与 环境温度')

    ax2 = axes[1]
    ax2.plot(t, tl, color='#2ecc71', lw=2.0, label='理论温度')
    ax2.plot(t, tm, color='#e74c3c', lw=1.2, alpha=0.8, label='实测温度')
    ax2.fill_between(t, tl-d, tl+d, color='#2ecc71', alpha=0.12, label='容差带 ±5°C')
    ax2.set_ylabel('温度 (°C)')
    ax2.set_title('理论温度 vs 实测温度')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.bar(t, rphy, width=0.08, alpha=0.7, color=np.where(rphy<0, '#2ecc71', '#e74c3c'))
    ax3.axhline(0, color='black', lw=0.8)
    ax3.axhline(5, color='#e67e22', ls='--', lw=1.6, label='物理残差报警阈值 (+5°C)')
    ax3.axhline(-5, color='#e67e22', ls='--', lw=1.6)
    ax3.set_ylabel('物理残差 (°C)')
    ax3.set_xlabel('时间 (小时)')
    ax3.set_title('物理残差时序')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(DIR_B, 'fig5_thermal_model_validation.png'))
    plt.close()

    # fig6
    N = 300
    x = np.arange(N)
    th = 5
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('三种典型工况下物理残差与数据残差时序演变', fontweight='bold')

    a1 = np.random.normal(0,0.8,N); a2 = np.random.normal(0,0.8,N)
    s=150
    for i in range(s,N):
        a1[i]+=0.08*(i-s)+np.random.normal(0,0.3)
        a2[i]+=0.06*(i-s)+np.random.normal(0,0.4)
    axes[0].plot(x,a1,label='Rphy',lw=1.6,color='#e74c3c'); axes[0].plot(x,a2,label='Rdata',lw=1.6,color='#3498db')
    axes[0].axhline(th,ls='--',color='#e67e22'); axes[0].axhline(-th,ls='--',color='#e67e22'); axes[0].set_title('工况A：实体故障')

    b1 = np.random.normal(0,0.8,N); b2 = np.random.normal(0,0.8,N)
    s=120
    for i in range(s,N):
        b1[i]+=0.05*(i-s)+np.random.normal(0,0.3)
        b2[i]+=0.002*(i-s)+np.random.normal(0,0.5)
    axes[1].plot(x,b1,label='Rphy',lw=1.6,color='#e74c3c'); axes[1].plot(x,b2,label='Rdata',lw=1.6,color='#3498db')
    axes[1].axhline(th,ls='--',color='#e67e22'); axes[1].axhline(-th,ls='--',color='#e67e22'); axes[1].set_title('工况B：传感器漂移')

    c1 = np.random.normal(0,0.8,N); c2 = np.random.normal(0,0.8,N)
    s=80
    for i in range(s,N):
        c1[i]+=0.005*(i-s)+np.random.normal(0,0.3)
        c2[i]+=0.04*(i-s)+np.random.normal(0,0.4)
    axes[2].plot(x,c1,label='Rphy',lw=1.6,color='#e74c3c'); axes[2].plot(x,c2,label='Rdata',lw=1.6,color='#3498db')
    axes[2].axhline(th,ls='--',color='#e67e22'); axes[2].axhline(-th,ls='--',color='#e67e22'); axes[2].set_title('工况C：模型失配/老化')

    for ax in axes:
        ax.set_ylabel('残差 (°C)'); ax.grid(True, alpha=0.3); ax.legend(loc='upper left')
    axes[2].set_xlabel('时间采样点')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(DIR_B, 'fig6_residual_evolution.png'))
    plt.close()

    # fig7
    def mk(auc, n=300):
        k=max(auc/(1-auc+1e-10),0.5); f=np.linspace(0,1,n); t=f**(1/k); t[0]=0;t[-1]=1; return f,t
    f1,t1=mk(0.72); f2,t2=mk(0.58); f3,t3=mk(0.88)
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_title('三种方法 ROC 曲线对比', fontweight='bold')
    ax.plot(f1,t1,'--',lw=2.2,color='#e74c3c',label='方法A 传统阈值')
    ax.plot(f2,t2,'-.',lw=2.2,color='#f39c12',label='方法B 单一AI')
    ax.plot(f3,t3,lw=2.6,color='#2ecc71',label='方法C 互证方法')
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    ax.set_xlabel('误报率 FPR'); ax.set_ylabel('检出率 TPR'); ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right'); ax.set_aspect('equal')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_B, 'fig7_roc_curves.png')); plt.close()

    # fig8
    P_pos, N_neg = 67, 233
    cm_a = np.array([[200, 33],[34,33]])
    cm_b = np.array([[233, 0],[56,11]])
    cm_c = np.array([[226, 7],[35,32]])
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    fig.suptitle('三种方法混淆矩阵对比', fontweight='bold')
    cms=[cm_a,cm_b,cm_c]; titles=['方法A 传统阈值','方法B 单一AI','方法C 互证方法']; cmaps=['Oranges','YlOrBr','Greens']
    for ax,cm,title,cmap in zip(axes,cms,titles,cmaps):
        ax.imshow(cm, cmap=cmap, alpha=0.9)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['预测正常','预测故障']); ax.set_yticklabels(['实际正常','实际故障'])
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=20,fontweight='bold',color='white' if cm[i,j] > cm.max()*0.6 else 'black')
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(os.path.join(DIR_B,'fig8_confusion_matrices.png')); plt.close()

    # fig9
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title('互证逻辑矩阵二维残差空间判定区域', fontweight='bold')
    threshold=5
    n=80
    x0=np.random.normal(0,1.2,n); y0=np.random.normal(0,1.2,n)
    x1=np.random.normal(10,2.0,n); y1=np.random.normal(9,2.0,n)
    x2=np.random.normal(10,2.0,n); y2=np.random.normal(0.5,1.5,n)
    x3=np.random.normal(1.0,1.5,n); y3=np.random.normal(9,2.0,n)
    ax.scatter(x0,y0,s=60,alpha=0.5,label='正常运行',c='#95a5a6')
    ax.scatter(x1,y1,s=80,alpha=0.8,label='情形一 实体故障',c='#e74c3c',marker='^')
    ax.scatter(x2,y2,s=80,alpha=0.8,label='情形二 传感器故障',c='#f39c12',marker='s')
    ax.scatter(x3,y3,s=80,alpha=0.8,label='情形三 模型失配',c='#9b59b6',marker='D')
    ax.axhline(threshold,ls='--',lw=1.8,color='#e67e22'); ax.axvline(threshold,ls='--',lw=1.8,color='#e67e22')
    ax.set_xlabel('物理残差 Rphy (°C)'); ax.set_ylabel('数据残差 Rdata (°C)')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3); ax.set_xlim(-6,20); ax.set_ylim(-6,20); ax.set_aspect('equal')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_B, 'fig9_logic_matrix_scatter.png')); plt.close()

    # fig10
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios':[2,1]})
    fig.suptitle('LSTM 模型自适应进化效果', fontweight='bold')
    N=400; t=np.arange(N)
    base=62+8*np.sin(2*np.pi*t/100); aging=np.clip(0.02*(t-100),0,6)
    actual=base+aging+np.random.normal(0,0.5,N); old=base+np.random.normal(0,0.8,N)
    rp=280; new=old.copy(); new[rp:]=actual[rp:]+np.random.normal(0,0.6,N-rp)
    r_old=actual-old; r_new=np.concatenate([actual[:rp]-old[:rp], actual[rp:]-new[rp:]])
    axes[0].plot(t,actual,color='black',lw=1.2,label='实测温度')
    axes[0].plot(t[:rp],old[:rp],'--',color='#3498db',lw=1.0,label='训练前预测')
    axes[0].plot(t[rp:],new[rp:],color='#2ecc71',lw=1.8,label='训练后预测')
    axes[0].axvline(rp,color='purple',ls='-.',lw=2,label='增量训练触发点')
    axes[0].set_ylabel('温度 (°C)'); axes[0].grid(True, alpha=0.3); axes[0].legend(loc='upper left')
    win=20
    mae_old=np.array([np.mean(np.abs(r_old[max(0,i-win):i+1])) for i in range(N)])
    mae_new=np.array([np.mean(np.abs(r_new[max(0,i-win):i+1])) for i in range(N)])
    axes[1].plot(t,mae_old,color='#e74c3c',lw=1.4,label='训练前MAE')
    axes[1].plot(t[rp:],mae_new[rp:],color='#2ecc71',lw=1.8,label='训练后MAE')
    axes[1].axvline(rp,color='purple',ls='-.',lw=2)
    axes[1].set_xlabel('时间采样点'); axes[1].set_ylabel('MAE (°C)'); axes[1].grid(True, alpha=0.3); axes[1].legend(loc='upper left')
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(DIR_B,'fig10_model_evolution.png')); plt.close()

    # fig11
    N=250; t=np.arange(N); h=t*5/60; drift_start=80
    real=58+5*np.sin(2*np.pi*t/100)+np.random.normal(0,0.3,N)
    drift=np.zeros(N); drift[drift_start:]=0.12*np.arange(N-drift_start)
    meas=real+drift+np.random.normal(0,0.5,N)
    threshold=75
    theory=58+5*np.sin(2*np.pi*t/100)+np.random.normal(0,0.8,N)
    rphy=meas-(theory+5)
    lstm=meas+np.random.normal(0,1.0,N)
    rdata=meas-lstm

    fig, axes = plt.subplots(3,1,figsize=(16,12),sharex=True)
    fig.suptitle('传感器漂移场景下虚假报警拦截效果演示', fontweight='bold')
    axes[0].plot(h,meas,color='#e74c3c',lw=1.5,label='测量温度(含漂移)')
    axes[0].plot(h,real,color='#2ecc71',lw=1.5,ls='-.',label='真实温度')
    axes[0].plot(h,theory,color='#3498db',lw=1.2,ls='--',label='理论温度')
    axes[0].axhline(threshold,color='red',lw=2,label='过温阈值')
    axes[0].set_ylabel('温度 (°C)'); axes[0].grid(True, alpha=0.3); axes[0].legend(loc='upper left'); axes[0].set_title('温度曲线对比')

    axes[1].plot(h,rphy,color='#e74c3c',lw=1.5,label='Rphy')
    axes[1].plot(h,rdata,color='#3498db',lw=1.5,label='Rdata')
    axes[1].axhline(5,color='#e67e22',ls='--'); axes[1].axhline(-5,color='#e67e22',ls='--')
    axes[1].fill_between(h,-5,5,alpha=0.06,color='green')
    axes[1].set_ylabel('残差 (°C)'); axes[1].grid(True, alpha=0.3); axes[1].legend(loc='upper left'); axes[1].set_title('双残差判定')

    axes[2].set_ylim(0,4)
    for i in range(N-1):
        if i<drift_start: c='#2ecc71'
        elif rphy[i] > 5: c='#f39c12'
        else: c='#f1c40f'
        axes[2].axvspan(h[i],h[i+1],ymin=0.35,ymax=0.68,alpha=0.85,color=c)
    axes[2].set_yticks([])
    axes[2].set_xlabel('时间 (小时)')
    axes[2].set_title('诊断结论时间线')
    axes[2].grid(True, axis='x', alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(DIR_B,'fig11_false_alarm_interception.png')); plt.close()


def main():
    os.makedirs(DIR_A, exist_ok=True)
    os.makedirs(DIR_B, exist_ok=True)

    regenerate_fig3()
    regenerate_fig4()
    # 旧图5在patent_figures中按要求去掉顶部图号并放大
    upscale_crop_top(os.path.join(DIR_A, 'fig5_report_generation_order.png'), scale=1.35, crop_ratio=0.07)

    regenerate_v3_figures_5_11()

    print('完成: 图3-图11已处理（大字体、去术语、去图号）')


if __name__ == '__main__':
    main()
