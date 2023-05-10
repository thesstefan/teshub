import { useEffect } from 'react';
import './RangeInput.css'

type RangeInputProps = {
    id: string;
    min?: number;
    max?: number;
    step?: number;
    value?: number;
    color?: string;
    defaultValue?: number;
    onChange?: (value: number) => void;
};

function RangeInput(props: RangeInputProps) {
    const { id, min = 0, max = 100, color = 'white', step = null, defaultValue, value, onChange } = props;

    useEffect(() => {
        const input = document.getElementById(id) as HTMLInputElement;
        input.style.setProperty('--color', color)
        input.style.setProperty('--value', input.value);
        input.style.setProperty('--min', min.toString());
        input.style.setProperty('--max', max.toString());
        input.addEventListener('input', () => {
            input.style.setProperty('--value', input.value);
            onChange && onChange(Number(input.value));
        });
    }, [id, min, max, color, onChange]);

    return <input 
                type="range" 
                className='range-input slider-progress' 
                id={id} 
                min={min} 
                max={max} 
                step={step || (max - min) / 10} 
                defaultValue={defaultValue} 
                value={value}
            />;
}

export default RangeInput;
