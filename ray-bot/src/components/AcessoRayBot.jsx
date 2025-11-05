import React, { useState } from 'react';
import styles from './AcessoRayBot.module.css'; 
import LogoRay from '../assets/LogoRay.png'; 

const AcessoRayBot = ({ onAccessGranted }) => {
  const [accessCode, setAccessCode] = useState('');
  
  const handleSubmit = (event) => {
    event.preventDefault();
    if (accessCode.trim() === '12345') { // Código de exemplo
      onAccessGranted(true); 
    } else {
      alert("Código de acesso inválido! Tente '12345'."); 
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <img src={LogoRay} alt="RayBot Logo" className={styles.logo} />
        
        <h2 className={styles.welcome}>Welcome to RayBot</h2>
        <p className={styles.subtitle}>Enter your access code to continue</p>
        
        <form onSubmit={handleSubmit} className={styles.form}>
          <input
            type="password"
            placeholder="Enter access code"
            value={accessCode}
            onChange={(e) => setAccessCode(e.target.value)}
            className={`${styles.input} input`} // Aplica a classe global .input
            aria-label="Access Code"
            required
          />
          <button 
            type="submit" 
            className={`${styles.button} button`} // Aplica a classe global .button
            disabled={!accessCode.trim()}
          >
            Access RayBot
          </button>
        </form>
      </div>
    </div>
  );
};

export default AcessoRayBot;