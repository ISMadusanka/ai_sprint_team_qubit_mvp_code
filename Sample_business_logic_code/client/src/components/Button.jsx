import React from 'react';

function Button({ onClick, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled}>
      Start Process
    </button>
  );
}

export default Button;
