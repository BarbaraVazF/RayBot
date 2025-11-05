import React from 'react';
import styles from './SegundaTela.module.css';

const SegundaTela = ({ onLogout }) => {
  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h2 style={{color: 'var(--color-primary)'}}>🚀 Bem-vindo ao RayBot!</h2>
        <p style={{marginBottom: 'var(--spacing-4)'}}>Esta é a área do aplicativo. Aqui você pode começar a construir sua interface de chatbot/ferramenta seguindo o Design System.</p>
        <button 
            onClick={onLogout} 
            className="button" // Reutiliza o estilo global do botão
        >
            Sair e Voltar
        </button>
      </div>
    </div>
  );
};

export default SegundaTela;