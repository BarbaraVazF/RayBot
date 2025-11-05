import React, { useState } from 'react';
import AcessoRayBot from './components/AcessoRayBot';
import SegundaTela from './components/SegundaTela';

// Você pode remover o "import './App.css';" se não estiver usando.

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Função para transição de tela
  const handleAccess = (status) => {
    setIsLoggedIn(status);
  };

  return (
    // Usa um fragmento
    <>
      {isLoggedIn ? (
        <SegundaTela onLogout={() => handleAccess(false)} />
      ) : (
        <AcessoRayBot onAccessGranted={handleAccess} />
      )}
    </>
  );
}

export default App;